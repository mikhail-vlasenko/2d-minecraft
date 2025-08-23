use std::cmp::{max};
use serde::{Serialize, Deserialize};
use crate::auxiliary::animations::{ProjectileType, TileAnimationType};
use crate::character::acting_with_speed::ActingWithSpeed;
use crate::map_generation::mobs::a_star::can_step;
use crate::character::player::Player;
use crate::map_generation::field::{AbsolutePos, Field, RelativePos};
use crate::map_generation::field::DIRECTIONS;
use crate::map_generation::mobs::mob_kind::{BANELING_EXPLOSION_PWR, BANELING_EXPLOSION_RAD, MobKind, MobState, ZERGLING_ATTACK_RANGE};
use crate::map_generation::mobs::mob_kind::MobKind::{Baneling, GelatinousCube, Zergling};
use crate::SETTINGS;


#[derive(PartialEq, Copy, Clone, Serialize, Deserialize, Debug)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: usize,
}

impl Position {
    pub fn new(xy: AbsolutePos, field: &Field) -> Self {
        let x = xy.0;
        let y = xy.1;
        let z = field.len_at(xy);
        Position { x, y, z }
    }

    pub fn manhattan_distance(&self, other: &Position) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}

impl From<(i32, i32)> for Position {
    fn from(xy: (i32, i32)) -> Self {
        Position { x: xy.0, y: xy.1, z: 0 }
    }
}

impl Into<(i32, i32)> for Position {
    fn into(self) -> (i32, i32) {
        (self.x, self.y)
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct Mob {
    pub pos: Position,
    rotation: i32,
    kind: MobKind,
    hp: i32,
    /// when this reaches 1, the mob is eligible to act
    speed_buffer: f32,
    state: MobState,
}

impl Mob {
    pub fn new(pos: Position, kind: MobKind) -> Self {
        Mob {
            pos,
            rotation: 0,
            kind,
            hp: kind.get_max_hp(),
            speed_buffer: 0.,
            state: MobState::default(),
        }
    }

    pub fn get_kind(&self) -> &MobKind {
        &self.kind
    }

    /// Makes a step, or a melee hit, if it is possible.
    /// Returns whether an action was made.
    fn step(&mut self, field: &Field, player: &mut Player, delta: (i32, i32),
            min_loaded: (i32, i32), max_loaded: (i32, i32)) -> bool {
        let new_pos = (self.pos.x + delta.0, self.pos.y + delta.1);

        if min_loaded.0 <= new_pos.0 && new_pos.0 <= max_loaded.0 &&
            min_loaded.1 <= new_pos.1 && new_pos.1 <= max_loaded.1 &&
            can_step(field, (self.pos.x, self.pos.y), new_pos) {

            self.set_rotation(Self::coords_to_rotation(delta));
            let (player_x, player_y) = player.xy();
            if player_x == new_pos.0 && player_y == new_pos.1 {
                player.receive_damage(self.kind.get_melee_damage());
            } else {
                self.pos.x = new_pos.0;
                self.pos.y = new_pos.1;
                self.land(field);
            }
            return true;
        }
        false
    }

    pub fn land(&mut self, field: &Field) {
        self.pos.z = field.len_at((self.pos.x, self.pos.y));
    }

    /// Moves in a random direction, not walking out of loaded chunks
    fn random_step(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let number: usize = rand::random();
        let direction_idx = number % 4;
        self.step(field, player, DIRECTIONS[direction_idx], min_loaded, max_loaded);
    }

    fn step_relative_to_player(&mut self, field: &mut Field, player: &mut Player,
                               min_loaded: (i32, i32), max_loaded: (i32, i32), towards: bool) {
        let (player_x, player_y) = player.xy();
        let directions = if towards {
            ((player_x - self.pos.x).signum(), (player_y - self.pos.y).signum())
        } else {
            // otherwise, move away from player
            ((self.pos.x - player_x).signum(), (self.pos.y - player_y).signum())
        };

        let mut possible = Vec::new();
        let mut good = Vec::new();
        for dir in DIRECTIONS {
            if can_step(field, (self.pos.x, self.pos.y),
                        (self.pos.x + dir.0, self.pos.y + dir.1)) {
                possible.push(dir);
                if (dir.0 == directions.0 && dir.0 != 0) || (dir.1 == directions.1 && dir.1 != 0) {
                    good.push(dir);
                }
            }
        }
        let idx: usize = rand::random();
        if good.len() > 0 {
            self.step(field, player, good[idx % good.len()], min_loaded, max_loaded);
        } else if possible.len() > 0 {
            self.step(field, player, possible[idx % possible.len()], min_loaded, max_loaded);
        }
    }

    fn check_baneling_explosion(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let (player_x, player_y) = player.xy();
        let directions = ((player_x - self.pos.x).signum(), (player_y - self.pos.y).signum());

        // baneling will explode if the shortest path to player is blocked, or it is next to the player
        let this_height = field.len_at((self.pos.x, self.pos.y));
        let vertical_cant_go = directions.0 == 0 ||
            field.len_at((self.pos.x + directions.0, self.pos.y)) > this_height + 1;
        let horizontal_cant_go = directions.1 == 0 ||
            field.len_at((self.pos.x, self.pos.y + directions.1)) > this_height + 1;

        let next_to_player = self.pos.x + directions.0 == player_x && self.pos.y + directions.1 == player_y;
        let visible = (max((player_x - self.pos.x).abs(),
                           (player_y - self.pos.y).abs()) as usize) <= SETTINGS.read().unwrap().window.render_distance as usize;

        if (vertical_cant_go && horizontal_cant_go  || next_to_player) && visible {
            let (a_star_directions, len) = field.full_pathing(
                (self.pos.x, self.pos.y),
                (player_x, player_y),
                (player_x, player_y),
                Some(2)
            );
            if a_star_directions == (0, 0) || next_to_player {
                // no route found with tiny detour or already near player
                field.explosion((self.pos.x, self.pos.y),
                                BANELING_EXPLOSION_RAD,
                                BANELING_EXPLOSION_PWR,
                                player);
                // baneling dies
                self.receive_damage(self.kind.get_max_hp());
                return;
            }
        }
    }
    
    fn jump_step(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let (player_x, player_y) = player.xy();
        let dist = (player_x - self.pos.x).abs() + (player_y - self.pos.y).abs();
        let mut new_pos = (self.pos.x, self.pos.y);
        if dist <= self.kind.jump_distance() {
            // jump on the player
            new_pos = (player_x, player_y);
            player.receive_damage(self.kind.get_melee_damage());
        } else {
            // jump in the direction of the player
            let dir = ((player_x - self.pos.x).signum(), (player_y - self.pos.y).signum());
            for _ in 0..self.kind.jump_distance() {
                if rand::random() {
                    new_pos.0 += dir.0;
                } else {
                    new_pos.1 += dir.1;
                }
            }
        }
        if field.is_occupied(new_pos) {
            return;
        }
        if self.kind == GelatinousCube {
            field.animations_buffer.add_projectile_animation(
                ProjectileType::GelatinousCube, (self.pos.x, self.pos.y), new_pos
            );
        }
        self.pos.x = new_pos.0;
        self.pos.y = new_pos.1;
        self.land(field);
    }

    pub fn update_state(&mut self, field: &Field, player_pos: &Position) {
        if self.kind == Zergling && self.state == MobState::Searching {
            let dist = self.pos.manhattan_distance(player_pos);
            if dist <= ZERGLING_ATTACK_RANGE {
                // check if there are at least 2 other zerglings near the player
                let indices = field.mob_indices(player_pos.clone().into(), Zergling);
                // indices will never have the current mob because it is neither in stray nor in chunk
                let mut close_count = 0;
                for ind in indices {
                    let pos: RelativePos = ind.0;
                    let dist = pos.0.abs() + pos.1.abs();
                    if dist <= ZERGLING_ATTACK_RANGE {
                        close_count += 1;
                    }
                }
                if close_count >= 2 {
                    self.state = MobState::Attacking;
                }
            }
        }
    }

    pub fn receive_damage(&mut self, damage: i32) -> bool {
        self.hp -= damage;
        if self.hp <= 0{
            return true;
        }
        false
    }

    pub fn is_alive(&self) -> bool {
        self.hp > 0
    }

    fn coords_to_rotation(coords: (i32, i32)) -> i32 {
        match coords {
            (-1, 0) => 0,
            (0, -1) => 1,
            (1, 0) => 2,
            (0, 1) => 3,
            _ => panic!("bad coords {}, {}", coords.0, coords.1)
        }
    }

    fn set_rotation(&mut self, rotation: i32) {
        self.rotation = rotation
    }

    /// Computes rotation as an integer between 0 and 4, where 0 is up, and 3 is right.
    pub fn get_rotation(&self) -> u32 {
        let modulus = self.rotation % 4;
        if modulus >= 0 {
            modulus as u32
        } else {
            (modulus + 4) as u32
        }
    }
    
    pub fn get_hp_share(&self) -> f32 {
        self.hp as f32 / self.kind.get_max_hp() as f32
    }
    
    pub fn is_channeling(&self) -> bool {
        if let MobState::Channeling(_) = self.state {
            return true;
        }
        false
    }
}

impl ActingWithSpeed for Mob {
    /// Performs a single turn on the mob.
    /// The mob cant wander out of loaded chunks.
    ///
    /// # Arguments
    /// * `field` - the field the mob is on
    /// * `player` - the player
    /// * `min_loaded` - the minimum loaded coordinate
    /// * `max_loaded` - the maximum loaded coordinate
    fn act(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        self.update_state(field, player.get_position());
        if let MobState::Channeling(turns) = self.state {
            if turns != 0 {
                self.state = MobState::Channeling(turns - 1);
                // end turn immediately
                return;
            }
            if self.kind == GelatinousCube {
                // cube jumps
                self.state = MobState::default();
                self.jump_step(field, player, min_loaded, max_loaded);
                return;
            }
        }

        let dist = self.pos.manhattan_distance(player.get_position());

        if dist == 0 {
            // deal damage because mob stands on the player
            player.receive_damage(self.kind.get_melee_damage());
            return;
        }

        if dist <= field.get_towards_player_radius() && self.kind == Baneling {
            // a bane can explode if the path is blocked
            self.check_baneling_explosion(field, player, min_loaded, max_loaded);
        }

        // hostile mobs within smaller range use optimal pathing. Banelings always go head on
        if self.kind.hostile() && dist <= field.get_a_star_radius() {
            if self.kind == Zergling && self.state != MobState::Attacking && dist + 2 < ZERGLING_ATTACK_RANGE {
                // zerglings that are not attacking will not go close to player
                self.step_relative_to_player(field, player, min_loaded, max_loaded, false);
                return;
            }

            // within a* range, so do full path search
            let (direction, _) = field.full_pathing(
                (self.pos.x, self.pos.y),
                player.xy(),
                player.xy(),
                None
            );
            if self.kind == GelatinousCube && direction == (0, 0) && self.kind.jump_distance() >= dist {
                // route not found, so jump
                self.state = MobState::Channeling(2);
                return;
            }
            if direction != (0, 0) {
                self.step(field, player, direction, min_loaded, max_loaded);
                return;
            }
        }
        if self.kind.hostile() && dist <= field.get_towards_player_radius() {
            self.step_relative_to_player(field, player, min_loaded, max_loaded, true);
            return;
        }
        // not hostile, or too far away, so just wander
        self.random_step(field, player, min_loaded, max_loaded);
    }
    
    fn get_speed(&self) -> f32 {
        self.kind.speed()
    }
    fn get_speed_buffer(&self) -> f32 {
        self.speed_buffer
    }
    fn add_to_speed_buffer(&mut self, amount: f32) {
        self.speed_buffer += amount;
    }
    fn decrement_speed_buffer(&mut self) {
        self.speed_buffer -= 1.0;
    }
}
