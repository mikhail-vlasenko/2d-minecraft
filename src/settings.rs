#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(dead_code)]

use std::borrow::Cow;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct Settings {
    pub field: _Config__field,
    pub mobs: _Config__mobs,
    pub pathing: _Config__pathing,
    pub player: _Config__player,
    pub save_folder: Cow<'static, str>,
    pub window: _Config__window,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__field {
    pub chunk_size: i32,
    pub from_test_chunk: bool,
    pub generation: _Config__field__generation,
    pub loading_distance: i32,
    pub map_radius: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__field__generation {
    pub diamond_proba: f32,
    pub iron_proba: f32,
    pub rock_proba: f32,
    pub structures: _Config__field__generation__structures,
    pub tree_proba: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__field__generation__structures {
    pub robot_proba: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs {
    pub spawning: _Config__mobs__spawning,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs__spawning {
    pub base_day_amount: i32,
    pub base_night_amount: i32,
    pub increase_amount_every: i32,
    pub initial_hostile_per_chunk: f32,
    pub max_mobs_on_chunk: i32,
    pub probabilities: _Config__mobs__spawning__probabilities,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__mobs__spawning__probabilities {
    pub bane: f32,
    pub ling: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__pathing {
    pub a_star_radius: i32,
    pub default_detour: i32,
    pub towards_player_radius: _Config__pathing__towards_player_radius,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__pathing__towards_player_radius {
    pub early: i32,
    pub red_moon: i32,
    pub usual: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__player {
    pub arrow_break_chance: f32,
    pub cheating_start: bool,
    pub max_hp: i32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_camel_case_types)]
pub struct _Config__window {
    pub height: i32,
    pub tiles_per_row: i32,
    pub width: i32,
}

pub const DEFAULT_SETTINGS: Settings = Settings {
    field: _Config__field {
        chunk_size: 16,
        from_test_chunk: false,
        generation: _Config__field__generation {
            diamond_proba: 0.05,
            iron_proba: 0.2,
            rock_proba: 0.05,
            structures: _Config__field__generation__structures {
                robot_proba: 0.1,
            },
            tree_proba: 0.07,
        },
        loading_distance: 5,
        map_radius: 64,
    },
    mobs: _Config__mobs {
        spawning: _Config__mobs__spawning {
            base_day_amount: 2,
            base_night_amount: 5,
            increase_amount_every: 3,
            initial_hostile_per_chunk: 0.2,
            max_mobs_on_chunk: 3,
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
    player: _Config__player {
        arrow_break_chance: 0.3,
        cheating_start: true,
        max_hp: 100,
    },
    save_folder: Cow::Borrowed("game_saves"),
    window: _Config__window {
        height: 1600,
        tiles_per_row: 17,
        width: 1600,
    },
};

#[cfg(debug_assertions)]
impl Settings {
    pub fn load() -> Cow<'static, Self> {
        let filepath = concat!(env!("CARGO_MANIFEST_DIR"), "/settings.yaml");
        Self::load_from(filepath.as_ref()).expect("Failed to load Settings.")
    }

    pub fn load_from(filepath: &::std::path::Path) -> Result<Cow<'static, Self>, Box<dyn ::std::error::Error>> {
        let file_contents = ::std::fs::read_to_string(filepath)?;
        let result: Self = ::serde_yaml::from_str(&file_contents)?;
        Ok(Cow::Owned(result))
    }
}

#[cfg(not(debug_assertions))]
impl Settings {
    #[inline(always)]
    pub fn load() -> Cow<'static, Self> {
        Cow::Borrowed(&DEFAULT_SETTINGS)
    }

    #[inline(always)]
    pub fn load_from(_: &::std::path::Path) -> Result<Cow<'static, Self>, Box<dyn ::std::error::Error>> {
        Ok(Cow::Borrowed(&DEFAULT_SETTINGS))
    }
}
