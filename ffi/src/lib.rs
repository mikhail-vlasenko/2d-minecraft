use std::os::raw::c_char;
use std::sync::Mutex;
use interoptopus::{ffi_function, Inventory, InventoryBuilder, function};
use interoptopus::{Error, Interop};
use lazy_static::lazy_static;
use game_logic::auxiliary::actions::Action;

use crate::game_state::GameState;

pub mod game_state;
pub mod observation;
pub mod ffi_config;


lazy_static! {
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

#[ffi_function]
#[no_mangle]
pub extern "C" fn num_actions() -> i32 {
    let mut max = 0;
    while let Ok(_) = Action::try_from(max) {
        max += 1;
    }
    max
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn action_name(action: i32) -> *mut c_char {
    let action = Action::try_from(action).unwrap();
    let name = format!("{:?}", action);
    let c_str = std::ffi::CString::new(name).unwrap();
    c_str.into_raw()
}

pub fn ffi_inventory() -> Inventory {
    InventoryBuilder::new()
        .register(function!(reset))
        .register(function!(step))
        .register(function!(get_observation))
        .register(function!(num_actions))
        .register(function!(action_name))
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


#[test]
fn num_inventory_items() {
    use crate::observation::INVENTORY_SIZE;
    use game_logic::crafting::storable::ALL_STORABLES;
    println!("Length of inventory: {}", ALL_STORABLES.len());
    assert_eq!(INVENTORY_SIZE, ALL_STORABLES.len(), "change INVENTORY_SIZE manually");
}
