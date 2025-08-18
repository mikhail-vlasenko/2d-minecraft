use std::mem::swap;
use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;
use crate::character::player::Player;
use crate::map_generation::field::{AbsolutePos, Field};

pub trait AnimationInfo {
    fn get_num_frames(&self) -> u32;
    /// How many updates should pass before the frame is switched
    fn updates_per_frame(&self) -> u32;
    /// A continuous animation runs in a loop and gets dropped when player makes a turn
    fn continuous(&self) -> bool;
}

#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum TileAnimationType {
    RedHit,
    YellowHit,
    Channelling,
}

impl AnimationInfo for TileAnimationType {
    fn get_num_frames(&self) -> u32 {
        match self {
            TileAnimationType::RedHit => 14,
            TileAnimationType::YellowHit => 14,
            TileAnimationType::Channelling => 10,
        }
    }
    
    fn updates_per_frame(&self) -> u32 {
        match self {
            TileAnimationType::RedHit => 3,
            TileAnimationType::YellowHit => 1,
            TileAnimationType::Channelling => 5,
        }
    }
    
    fn continuous(&self) -> bool {
        match self {
            TileAnimationType::RedHit => false,
            TileAnimationType::YellowHit => false,
            TileAnimationType::Channelling => true,
        }
    }
}

/// Gives an animation type for an event
impl TileAnimationType {
    pub fn mining() -> Self {
        TileAnimationType::YellowHit
    }

    pub fn receive_damage() -> Self {
        TileAnimationType::RedHit
    }
}

/// An animation that plays only within one tile
#[derive(PartialEq, Copy, Clone, Serialize, Deserialize, Debug)]
pub struct TileAnimation {
    animation_type: TileAnimationType,
    pos: AbsolutePos,
    frame: u32,
    /// updates since the last frame switch
    unapplied_updates: i32,
}

impl TileAnimation {
    pub fn new(animation_type: TileAnimationType, pos: AbsolutePos) -> Self {
        Self {
            animation_type,
            pos,
            frame: 0,
            // when an animation is just added, it should not be updated immediately
            unapplied_updates: -1,
        }
    }
    pub fn get_pos(&self) -> &AbsolutePos {
        &self.pos
    }

    pub fn get_frame(&self) -> u32 {
        self.frame
    }

    pub fn get_num_frames(&self) -> u32 {
        self.animation_type.get_num_frames()
    }

    pub fn get_animation_type(&self) -> &TileAnimationType {
        &self.animation_type
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct AnimationManager {
    tile_animations: Vec<TileAnimation>,
    projectile_animations: Vec<ProjectileAnimation>,
    enabled: bool,
}

impl AnimationManager {
    pub fn new() -> Self {
        Self {
            tile_animations: Vec::new(),
            projectile_animations: Vec::new(),
            enabled: false,
        }
    }
    
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn get_tile_animations(&self) -> &Vec<TileAnimation> {
        &self.tile_animations
    }

    pub fn get_projectile_animations(&self) -> &Vec<ProjectileAnimation> {
        &self.projectile_animations
    }

    pub fn add_tile_animation(&mut self, tile_animation: TileAnimation) {
        self.tile_animations.push(tile_animation);
    }

    pub fn add_projectile_animation(&mut self, projectile_animation: ProjectileAnimation) {
        self.projectile_animations.push(projectile_animation);
    }
    
    pub fn absorb_buffer(&mut self, buffer: &mut AnimationsBuffer) {
        self.tile_animations.extend(buffer.get_new_tile_animations());
        self.projectile_animations.extend(buffer.get_new_projectile_animations());
    }

    pub fn update(&mut self) {
        for animation in &mut self.tile_animations {
            animation.unapplied_updates += 1;
            if animation.unapplied_updates >= animation.animation_type.updates_per_frame() as i32 {
                animation.unapplied_updates = 0;
                animation.frame += 1;
            }
            // loop continuous animations
            if animation.frame == animation.animation_type.get_num_frames() && animation.animation_type.continuous() {
                animation.frame = 0;
            }
        }
        // remove ones with frame == num_frames
        self.tile_animations.retain(|animation| {
            animation.frame < animation.animation_type.get_num_frames()
        });

        for animation in &mut self.projectile_animations {
            animation.progress += animation.projectile_type.get_speed();
        }
        self.projectile_animations.retain(|animation| {
            animation.progress <= animation.get_distance()
        });
    }
    
    pub fn drop_continuous_animations(&mut self) {
        self.tile_animations.retain(|animation| {
            !animation.animation_type.continuous()
        });
    }
    
    pub fn add_channeling_animations(&mut self, field: &Field, player: &Player) {
        for mob_pos in field.conditional_close_mob_info(
            |mob| { (mob.pos.x, mob.pos.y) },
            player,
            |mob| { mob.is_channeling() } 
        ) {
            self.add_tile_animation(TileAnimation::new(TileAnimationType::Channelling, mob_pos));
        }
    }
    
    pub fn clear(&mut self) {
        self.tile_animations.clear();
        self.projectile_animations.clear();
    }
}

#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum ProjectileType {
    Arrow,
    GelatinousCube,
}

impl ProjectileType {
    pub fn get_speed(&self) -> f32 {
        match self {
            ProjectileType::Arrow => 0.1,
            ProjectileType::GelatinousCube => 0.05,
        }
    }
}

#[derive(PartialEq, Copy, Clone, Serialize, Deserialize, Debug)]
pub struct ProjectileAnimation {
    source: AbsolutePos,
    target: AbsolutePos,
    // as a distance in tiles
    progress: f32,
    distance: f32,
    projectile_type: ProjectileType,
}

impl ProjectileAnimation {
    pub fn new(projectile_type: ProjectileType, source: AbsolutePos, target: AbsolutePos) -> Self {
        let distance = ((source.0 - target.0).pow(2) as f32 + (source.1 - target.1).pow(2) as f32).sqrt();
        Self {
            source,
            target,
            progress: 0.0,
            distance,
            projectile_type,
        }
    }
    fn get_distance(&self) -> f32 {
        self.distance
    }

    pub fn get_position(&self) -> (f32, f32) {
        let source = (self.source.0 as f32 + 0.5, self.source.1 as f32 + 0.5);
        let target = (self.target.0 as f32 + 0.5, self.target.1 as f32 + 0.5);
        let x = source.0 + (target.0 - source.0) * self.progress / self.distance;
        let y = source.1 + (target.1 - source.1) * self.progress / self.distance;
        (x, y)
    }

    pub fn get_relative_position(&self, player: &Player) -> (f32, f32) {
        let (x, y) = self.get_position();
        (x - player.x as f32, y - player.y as f32)
    }

    pub fn get_rotation(&self) -> f32 {
        let source = (self.source.0 as f32 + 0.5, self.source.1 as f32 + 0.5);
        let target = (self.target.0 as f32 + 0.5, self.target.1 as f32 + 0.5);
        let dx = -(target.0 - source.0);
        let dy = target.1 - source.1;
        -dy.atan2(dx)
    }
    
    pub fn get_projectile_type(&self) -> &ProjectileType {
        &self.projectile_type
    }
}

#[derive(PartialEq, Clone, Serialize, Deserialize, Debug)]
pub struct AnimationsBuffer {
    new_tile_animations: Vec<TileAnimation>,
    new_projectile_animations: Vec<ProjectileAnimation>,
}

impl AnimationsBuffer {
    pub fn new() -> Self {
        Self {
            new_tile_animations: Vec::new(),
            new_projectile_animations: Vec::new(),
        }
    }

    /// Returns the list and clears it.
    pub fn get_new_tile_animations(&mut self) -> Vec<TileAnimation> {
        let mut res = Vec::new();
        swap(&mut res, &mut self.new_tile_animations);
        res
    }

    /// Returns the list and clears it.
    pub fn get_new_projectile_animations(&mut self) -> Vec<ProjectileAnimation> {
        let mut res = Vec::new();
        swap(&mut res, &mut self.new_projectile_animations);
        res
    }

    pub fn add_tile_animation(&mut self, animation_type: TileAnimationType, pos: AbsolutePos) {
        self.new_tile_animations.push(TileAnimation::new(animation_type, pos));
    }

    pub fn add_projectile_animation(&mut self, projectile_type: ProjectileType, source: AbsolutePos, target: AbsolutePos) {
        self.new_projectile_animations.push(ProjectileAnimation::new(projectile_type, source, target));
    }
    
    pub fn clear(&mut self) {
        self.new_tile_animations.clear();
        self.new_projectile_animations.clear();
    }
}
