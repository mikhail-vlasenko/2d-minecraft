use crate::character::p2p_interactions::InteractionStore;
use crate::map_generation::field::Field;
use crate::map_generation::mobs::mob::Position;

pub trait ActingWithSpeed {
    /// Calls self.act() 0 or more times, depending on the actor's speed.
    fn act_with_speed(&mut self, field: &mut Field, player_pos: &Position, min_loaded: (i32, i32), max_loaded: (i32, i32)) -> InteractionStore {
        let mut interactions = Vec::new();
        self.add_to_speed_buffer(self.get_speed());
        while self.get_speed_buffer() >= 1.{
            interactions.extend(self.act(field, player_pos, min_loaded, max_loaded));
            self.decrement_speed_buffer();
        }
        interactions
    }
    fn act(&mut self, field: &mut Field, player_pos: &Position, min_loaded: (i32, i32), max_loaded: (i32, i32)) -> InteractionStore;
    fn get_speed(&self) -> f32;
    fn get_speed_buffer(&self) -> f32;
    fn add_to_speed_buffer(&mut self, amount: f32);
    fn decrement_speed_buffer(&mut self);
}
