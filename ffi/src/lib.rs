use std::sync::Mutex;
use crate::game_state::GameState;

mod game_state;
mod observation;
mod ffi_config;


lazy_static::lazy_static! {
    static ref STATE: Mutex<GameState> = Mutex::new(GameState::new());
}


#[no_mangle]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust!");
}

#[no_mangle]
pub extern "C" fn reset() {
    let mut state = STATE.lock().unwrap();
    *state = GameState::new();
}

#[no_mangle]
pub extern "C" fn step(action: i32) {
    let mut state = STATE.lock().unwrap();
    state.step_i32(action);
}

#[no_mangle]
pub extern "C" fn get_observation() -> observation::Observation {
    let state = STATE.lock().unwrap();
    state.get_observation()
}
