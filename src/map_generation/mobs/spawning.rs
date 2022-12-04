use rand::{random, Rng};
use crate::map_generation::chunk::Chunk;
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;


/// Takes into account how far away from center the player is
/// and how much time has passed since start.
/// (both increase mob difficulty)
pub fn spawn_hostile(chunk: & mut Chunk, x_chunk: i32, y_chunk: i32, game_time: f32) {
    let size = chunk.get_size() as i32;
    let (x, y) = pick_tile(&size);
    
    let pos = Position {
        x: x + size * x_chunk,
        y: y + size * y_chunk,
        z: chunk.len_at(x, y),
    };
    let kind = pick_mob_kind(x_chunk + y_chunk, game_time);
    let mob = Mob::new(pos, kind);
    chunk.add_mob(mob);
}

fn pick_tile(size: &i32) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(0..*size);
    let y = rng.gen_range(0..*size);
    (x, y)
}

fn pick_mob_kind(dist: i32, game_time: f32) -> MobKind {
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
