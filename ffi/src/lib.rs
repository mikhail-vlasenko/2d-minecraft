use std::sync::Mutex;
use interoptopus::{ffi_function, Inventory, InventoryBuilder, function};
use interoptopus::{Error, Interop};

use crate::game_state::GameState;

mod game_state;
mod observation;
mod ffi_config;


lazy_static::lazy_static! {
    // todo: a vec to allow batch processing
    static ref STATE: Mutex<GameState> = Mutex::new(GameState::new());
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn reset() {
    let mut state = STATE.lock().unwrap();
    *state = GameState::new();
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn step(action: i32) {
    let mut state = STATE.lock().unwrap();
    state.step_i32(action);
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn get_observation() -> observation::Observation {
    let state = STATE.lock().unwrap();
    state.get_observation()
}

pub fn ffi_inventory() -> Inventory {
    InventoryBuilder::new()
        .register(function!(reset))
        .register(function!(step))
        .register(function!(get_observation))
        .inventory()
}


#[test]
fn interoptopus_bindings_cpython() -> Result<(), Error> {
    use interoptopus_backend_cpython::{Config, Generator};

    let library = ffi_inventory();

    Generator::new(Config::default(), library)
        .write_file("../python_wrapper/ffi_elements.py")?;

    Ok(())
}
