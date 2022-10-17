trait Component {}

pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}
impl Component for Position {}

pub enum MobKind {
    Zombie,
    Cow,
}
impl Component for MobKind {}

pub struct Mob {
    hp: i32,
}
impl Component for Mob {}

