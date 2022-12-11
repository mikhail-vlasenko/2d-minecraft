use std::cmp::{max, min};
use rand::Rng;
use crate::graphics::state::RENDER_DISTANCE;
use crate::map_generation::mobs::a_star::{AStar, can_step};
use crate::player::Player;
use crate::map_generation::field::Field;
use crate::map_generation::field::DIRECTIONS;
use crate::map_generation::mobs::mob_kind::{BANELING_EXPLOSION_PWR, BANELING_EXPLOSION_RAD, MobKind};
use crate::map_generation::mobs::mob_kind::MobKind::Baneling;


#[derive(PartialEq, Copy, Clone, Debug)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: usize,
}

pub struct Mob {
    pub pos: Position,
    kind: MobKind,
    hp: i32,
    /// when this reaches 1, the mob is eligible to act
    speed_buffer: f32,
}

impl Mob {
    pub fn new(pos: Position, kind: MobKind) -> Self {
        Mob {
            pos,
            kind,
            hp: kind.get_max_hp(),
            speed_buffer: 0.,
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
            can_step(field, (self.pos.x, self.pos.y), new_pos, self.pos.z) {

            if player.x == new_pos.0 && player.y == new_pos.1 {
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

    pub fn act_with_speed(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        self.speed_buffer += self.kind.speed();
        while self.speed_buffer >= 1.{
            self.act(field, player, min_loaded, max_loaded);
            self.speed_buffer -= 1.;
        }
    }

    fn act(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let dist = (player.x - self.pos.x).abs() + (player.y - self.pos.y).abs();

        if dist <= field.get_towards_player_radius() && self.kind == Baneling {
            self.baneling_step(field, player, min_loaded, max_loaded);
        }

        // hostile mobs within smaller range use optimal pathing. Banelings always go head on
        if self.kind.hostile() && dist <= field.get_a_star_radius() {
            // within a* range, so do full path search
            let (direction, _) = field.full_pathing(
                (self.pos.x, self.pos.y),
                (player.x, player.y),
                (player.x, player.y),
                None
            );
            if direction != (0, 0) {
                self.step(field, player, direction, min_loaded, max_loaded);
            } else {
                self.step_towards_player(field, player, min_loaded, max_loaded);
            }
        }
        else if self.kind.hostile() && dist <= field.get_towards_player_radius() {
            self.step_towards_player(field, player, min_loaded, max_loaded);
        } else {
            self.random_step(field, player, min_loaded, max_loaded);
        }
    }

    /// Moves in a random direction, not walking out of loaded chunks
    fn random_step(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let number: usize = rand::random();
        let direction_idx = number % 4;
        self.step(field, player, DIRECTIONS[direction_idx], min_loaded, max_loaded);
    }

    fn step_towards_player(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let directions = ((player.x - self.pos.x).signum(), (player.y - self.pos.y).signum());

        let mut possible = Vec::new();
        let mut good = Vec::new();
        for dir in DIRECTIONS {
            if can_step(field, (self.pos.x, self.pos.y),
                        (self.pos.x + dir.0, self.pos.y + dir.1), self.pos.z) {
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

    fn baneling_step(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let directions = ((player.x - self.pos.x).signum(), (player.y - self.pos.y).signum());

        // baneling will explode if the shortest path to player is blocked, or it is next to the player
        let this_height = field.len_at((self.pos.x, self.pos.y));
        let vertical_cant_go = directions.0 == 0 ||
            field.len_at((self.pos.x + directions.0, self.pos.y)) > this_height + 1;
        let horizontal_cant_go = directions.1 == 0 ||
            field.len_at((self.pos.x, self.pos.y + directions.1)) > this_height + 1;

        let next_to_player = self.pos.x + directions.0 == player.x && self.pos.y + directions.1 == player.y;
        let visible = (max((player.x - self.pos.x).abs(),
                           (player.y - self.pos.y).abs()) as usize) <= RENDER_DISTANCE;

        if (vertical_cant_go && horizontal_cant_go  || next_to_player) && visible {
            let (a_star_directions, len) = field.full_pathing(
                (self.pos.x, self.pos.y),
                (player.x, player.y),
                (player.x, player.y),
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
}

pub fn mob_act_with_speed(mob: &mut Mob, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
    mob.act_with_speed(field, player, min_loaded, max_loaded)
}
