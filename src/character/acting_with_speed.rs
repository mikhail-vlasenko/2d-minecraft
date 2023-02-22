use crate::character::player::Player;
use crate::map_generation::field::Field;

pub trait ActingWithSpeed {
    /// Calls self.act() 0 or more times, depending on the actor's speed.
    fn act_with_speed(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32)) {
        self.add_to_speed_buffer(self.get_speed());
        while self.get_speed_buffer() >= 1.{
            self.act(field, player, min_loaded, max_loaded);
            self.decrement_speed_buffer();
        }
    }
    fn act(&mut self, field: &mut Field, player: &mut Player, min_loaded: (i32, i32), max_loaded: (i32, i32));
    fn get_speed(&self) -> f32;
    fn get_speed_buffer(&self) -> f32;
    fn add_to_speed_buffer(&mut self, amount: f32);
    fn decrement_speed_buffer(&mut self);
}