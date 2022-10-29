use std::cmp::{max, min};
use rand::Rng;
use strum_macros::EnumIter;
use crate::map_generation::mobs::a_star::AStar;
use crate::{Field, Player};


pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
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

    fn step(&mut self, delta: (i32, i32), min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let new_pos = (self.pos.x + delta.0, self.pos.y + delta.1);
        if min_loaded.0 <= new_pos.0 && new_pos.0 <= max_loaded.0 &&
            min_loaded.1 <= new_pos.1 && new_pos.1 <= max_loaded.1 {
            self.pos.x = new_pos.0;
            self.pos.y = new_pos.1;
        }
    }

    pub fn act(&mut self, field: &Field, player: &Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        let mut rng = rand::thread_rng();
        let mut pathing = AStar::new(field);
        let direction = pathing.a_star(field, (self.pos.x, self.pos.y), (player.x, player.y));
        self.step(direction, min_loaded, max_loaded);
    }
}
