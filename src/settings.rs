#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(dead_code)]

use std::borrow::Cow;

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct Settings {
    pub field: _Config__field,
    pub mobs: _Config__mobs,
    pub pathing: _Config__pathing,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__field {
    pub generation: _Config__field__generation,
    pub loading_distance: i32,
    pub render_distance: i32,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__field__generation {
    pub diamond_proba: f32,
    pub iron_proba: f32,
    pub rock_proba: f32,
    pub tree_proba: f32,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs {
    pub spawning: _Config__mobs__spawning,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs__spawning {
    pub base_day_amount: i32,
    pub base_night_amount: i32,
    pub probabilities: _Config__mobs__spawning__probabilities,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs__spawning__probabilities {
    pub bane: f32,
    pub ling: f32,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__pathing {
    pub a_star_radius: i32,
    pub default_detour: i32,
    pub towards_player_radius: _Config__pathing__towards_player_radius,
}

#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct _Config__pathing__towards_player_radius {
    pub early: i32,
    pub red_moon: i32,
    pub usual: i32,
}

pub const SETTINGS: Settings = Settings {
    field: _Config__field {
        generation: _Config__field__generation {
            diamond_proba: 0.05,
            iron_proba: 0.2,
            rock_proba: 0.05,
            tree_proba: 0.07,
        },
        loading_distance: 5,
        render_distance: 8,
    },
    mobs: _Config__mobs {
        spawning: _Config__mobs__spawning {
            base_day_amount: 2,
            base_night_amount: 5,
            probabilities: _Config__mobs__spawning__probabilities {
                bane: 0.3,
                ling: 0.2,
            },
        },
    },
    pathing: _Config__pathing {
        a_star_radius: 20,
        default_detour: 10,
        towards_player_radius: _Config__pathing__towards_player_radius {
            early: 35,
            red_moon: 55,
            usual: 45,
        },
    },
};
