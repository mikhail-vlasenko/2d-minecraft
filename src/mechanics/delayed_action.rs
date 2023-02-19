use crate::player::Player;

pub struct DelayedAction<T: ?Sized> {
    pub duration: u32,
    /// function to call at after the duration. i.e. remove an effect
    pub callback: fn(&mut Player, &T) -> (),
    pub argument: Box<T>,
}

impl <T: ?Sized> DelayedAction<T> {
    pub fn new(duration: u32, callback: fn(&mut Player, &T) -> (), argument: Box<T>) -> Self {
        Self {
            duration,
            callback,
            argument,
        }
    }

    /// Counts down the duration and
    /// returns true if the action is done and the instance can be removed.
    pub fn check(&mut self) -> bool {
        if self.duration == 0 {
            true
        } else {
            self.duration -= 1;
            false
        }
    }
}
