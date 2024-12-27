use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::distributions::Distribution;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::SETTINGS;


/// Takes into account how far away from center the player is
/// and how much time has passed since start.
/// (both increase mob difficulty)
pub fn create_mob(chunk: &mut Chunk, chunk_pos: (i32, i32), tile: (i32, i32), game_time: f32, hostile: bool) {
    let pos = Position {
        x: tile.0,
        y: tile.1,
        z: chunk.len_at(tile.0, tile.1),
    };
    let kind = if hostile {
        pick_hostile_kind(chunk_pos.0 + chunk_pos.1, game_time)
    } else {
        pick_friendly_kind()
    };
    let mob = Mob::new(pos, kind);
    chunk.add_mob(mob);
}

fn pick_tile(size: &i32) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(0..*size);
    let y = rng.gen_range(0..*size);
    (x, y)
}

fn pick_hostile_kind(dist: i32, game_time: f32) -> MobKind {
    let spawn_settings = SETTINGS.read().unwrap().mobs.spawning.clone();
    let spawn_hard_mobs = game_time > 100. * (spawn_settings.hard_mobs_since - 1) as f32;

    let mob_kinds = vec![MobKind::Zombie, MobKind::Zergling, MobKind::Baneling, MobKind::GelatinousCube];
    let mut weights = vec![
        0.,
        spawn_settings.probabilities.ling,
        if spawn_hard_mobs { spawn_settings.probabilities.bane } else { 0. },
        if spawn_hard_mobs { spawn_settings.probabilities.gelatinous_cube } else { 0. },
    ];
    // assign rest to zombie
    weights[0] = f32::max(1. - weights.iter().sum::<f32>(), 0.);
    let dist = WeightedIndex::new(&weights).unwrap();
    mob_kinds[dist.sample(&mut rand::thread_rng())]
}

fn pick_friendly_kind() -> MobKind {
    MobKind::Cow
}
