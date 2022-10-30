use rand::Rng;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::mobs::mob::{Mob, Position};
use crate::map_generation::mobs::mob_kind::MobKind;


pub fn spawn_hostile(chunk: & mut Chunk, x_chunk: i32, y_chunk: i32) {
    let size = chunk.get_size() as i32;
    let (x, y) = pick_tile(&size);
    
    let pos = Position {
        x: x + size * x_chunk,
        y: y + size * y_chunk,
        z: chunk.len_at(x, y),
    };
    let mob = Mob::new(pos, MobKind::Zombie);
    chunk.add_mob(mob);
}

fn pick_tile(chunk_size: &i32) -> (i32, i32) {
    let mut rng = rand::thread_rng();
    let x = rng.gen_range(0..*chunk_size);
    let y = rng.gen_range(0..*chunk_size);
    (x, y)
}
