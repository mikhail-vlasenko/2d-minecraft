use serde::{Deserialize, Serialize};
use strum_macros::EnumIter;
use crate::map_generation::field::AbsolutePos;

pub trait AnimationInfo {
    fn get_num_frames(&self) -> u32;
    /// How many updates should pass before the frame is switched
    fn updates_per_frame(&self) -> u32;
}

#[derive(PartialEq, Copy, Clone, Hash, EnumIter, Serialize, Deserialize, Debug)]
pub enum TileAnimationType {
    RedHit,
    YellowHit,
}

impl AnimationInfo for TileAnimationType {
    fn get_num_frames(&self) -> u32 {
        match self {
            TileAnimationType::RedHit => 14,
            TileAnimationType::YellowHit => 14,
        }
    }
    
    fn updates_per_frame(&self) -> u32 {
        match self {
            TileAnimationType::RedHit => 3,
            TileAnimationType::YellowHit => 1,
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
pub struct TileAnimation {
    animation_type: TileAnimationType,
    pos: AbsolutePos,
    frame: u32,
    /// updates since the last frame switch
    unapplied_updates: i32,
}

impl TileAnimation {
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

pub struct AnimationManager {
    tile_animations: Vec<TileAnimation>,
}

impl AnimationManager {
    pub fn new() -> Self {
        Self {
            tile_animations: Vec::new(),
        }
    }
    
    pub fn get_tile_animations(&self) -> &Vec<TileAnimation> {
        &self.tile_animations
    }
    
    pub fn add_tile_animation(&mut self, animation_type: TileAnimationType, pos: AbsolutePos) {
        self.tile_animations.push(TileAnimation {
            animation_type,
            pos,
            frame: 0,
            // when an animation is just added, it should not be updated immediately
            unapplied_updates: -1,
        });
    }
    
    pub fn update(&mut self) {
        for animation in &mut self.tile_animations {
            animation.unapplied_updates += 1;
            if animation.unapplied_updates >= animation.animation_type.updates_per_frame() as i32 {
                animation.unapplied_updates = 0;
                animation.frame += 1;
            }
        }
        // remove ones with frame == num_frames
        self.tile_animations.retain(|animation| {
            animation.frame < animation.animation_type.get_num_frames()
        });
    }
}


