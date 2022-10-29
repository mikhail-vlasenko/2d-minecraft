use std::cmp::{max, min};
use rand::Rng;
use strum_macros::EnumIter;
use crate::map_generation::mobs::a_star::{AStar, can_step};
use crate::{Field, Player};
use crate::map_generation::field::DIRECTIONS;


pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: usize,
}

#[derive(PartialEq, Copy, Clone, EnumIter, Debug)]
pub enum MobKind {
    Zombie,
    Cow,
}

pub struct Mob {
    pub pos: Position,
    kind: MobKind,
    hp: i32,
}

impl Mob {
    pub fn new(pos: Position, kind: MobKind, hp: i32,) -> Self {
        Mob {
            pos,
            kind,
            hp
        }
    }

    pub fn get_kind(&self) -> &MobKind {
        &self.kind
    }

    fn step(&mut self, field: &Field, delta: (i32, i32), min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let new_pos = (self.pos.x + delta.0, self.pos.y + delta.1);
        if min_loaded.0 <= new_pos.0 && new_pos.0 <= max_loaded.0 &&
            min_loaded.1 <= new_pos.1 && new_pos.1 <= max_loaded.1 &&
            can_step(field, (self.pos.x, self.pos.y), new_pos, self.pos.z) {
            self.pos.x = new_pos.0;
            self.pos.y = new_pos.1;
            self.land(field);
        }
    }

    pub fn land(&mut self, field: &Field) {
        self.pos.z = field.len_at(self.pos.x, self.pos.y);
    }

    pub fn act(&mut self, field: &mut Field, player: &Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let dist = min((player.x - self.pos.x).abs(), (player.y - self.pos.y).abs());
        if dist <= field.get_a_star_radius() {
            // within a* range, so do full path search
            let direction = field.full_pathing(
                (self.pos.x, self.pos.y),
                (player.x, player.y),
                (player.x, player.y)
            );
            self.step(field, direction, min_loaded, max_loaded);
        } else {
            // random valid move
            let mut rng = rand::thread_rng();
            let number: usize = rng.gen();
            let direction_idx = number % 4;
            self.step(field, DIRECTIONS[direction_idx], min_loaded, max_loaded);
        }
    }
}
