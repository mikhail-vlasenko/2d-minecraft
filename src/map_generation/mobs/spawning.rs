use rand::{random, Rng};
use crate::map_generation::chunk::Chunk;
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;


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
    return MobKind::Baneling;
    if game_time > 200. {
        // todo: banes
    }
    let rng: f32 = random();
    if rng > 0.8 {
        MobKind::Zergling
    } else {
        MobKind::Zombie
    }
}

fn pick_friendly_kind() -> MobKind {
    MobKind::Cow
}
