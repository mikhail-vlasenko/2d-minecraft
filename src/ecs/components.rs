use specs::prelude::*;
use specs_derive::*;
use wgpu::BindGroup;


#[derive(Component)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Component)]
pub struct Mob {
    hp: i32,
}

#[derive(Component)]
pub struct Monster {}

#[derive(Component)]
pub struct Renderable {
    pub bind_group: BindGroup,
}
