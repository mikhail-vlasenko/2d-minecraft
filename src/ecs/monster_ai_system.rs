use specs::prelude::*;
use crate::ecs::components::{Monster, Position};


struct MonsterAI;

impl<'a> System<'a> for MonsterAI {
    // These are the resources required for execution.
    // You can also define a struct and `#[derive(SystemData)]`,
    // see the `full` example.
    type SystemData = (WriteStorage<'a, Position>, WriteStorage<'a, Monster>);

    fn run(&mut self, data: Self::SystemData) {
        let (mut position, monster) = data;
        for (mut pos, _) in (&mut position, &monster).join() {
            pos.x += 1;
        }
    }
}
