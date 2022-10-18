use strum_macros::EnumIter;


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
}
