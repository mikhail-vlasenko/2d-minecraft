use std::os::raw::c_char;
use std::sync::Mutex;
use interoptopus::{ffi_function, Inventory, InventoryBuilder, function};
use interoptopus::{Error, Interop};
use lazy_static::lazy_static;
use game_logic::auxiliary::actions::Action;
use game_logic::SETTINGS;

use crate::game_state::GameState;
use crate::observation::{ActionMask, NUM_ACTIONS, Observation};

pub mod game_state;
pub mod observation;
pub mod ffi_config;


lazy_static! {
    static ref STATE: Mutex<Vec<GameState>> = Mutex::new(vec![GameState::new()]);
    static ref BATCH_SIZE: Mutex<usize> = Mutex::new(1);
    static ref CONNECTED: Mutex<Vec<bool>> = Mutex::new(vec![false]);
}

fn reset_all(batch_size: usize) {
    let mut new_states = Vec::new();
    for _ in 0..batch_size {
        new_states.push(GameState::new());
    }
    let mut state = STATE.lock().unwrap();
    *state = new_states;
}

/// Initializes batch_size games states.
/// Does a reset on all game states.
#[ffi_function]
#[no_mangle]
pub extern "C" fn set_batch_size(new_batch_size: i32) {
    reset_all(new_batch_size as usize);
    let mut batch_size = BATCH_SIZE.lock().unwrap();
    *batch_size = new_batch_size as usize;
    // set all envs as not connected
    let mut connected = CONNECTED.lock().unwrap();
    *connected = vec![false; new_batch_size as usize];
}

/// Finds an unconnected games state and returns its index.
/// Panics if all game states are connected.
#[ffi_function]
#[no_mangle]
pub extern "C" fn connect_env() -> i32 {
    let mut connected = CONNECTED.lock().unwrap();
    for (i, is_connected) in connected.iter_mut().enumerate() {
        if !*is_connected {
            *is_connected = true;
            return i as i32;
        }
    }
    panic!("All game states are connected. close_one() disconnects one game. set_batch_size() disconnects all games.");
}

/// Resets all game states.
/// Not advised. Especially since game states initialize ready to be stepped.
#[ffi_function]
#[no_mangle]
pub extern "C" fn reset() {
    reset_all(*BATCH_SIZE.lock().unwrap());
}

/// Resets the game state at the specified index.
#[ffi_function]
#[no_mangle]
pub extern "C" fn reset_one(index: i32) {
    let mut state = STATE.lock().unwrap();
    if let Some(game_state) = state.get_mut(index as usize) {
        game_state.reset();
    } else {
        panic!("Index {} out of bounds for batch size {}", index, state.len());
    }
}

/// Steps the game state at the specified index with the given action.
///
/// # Arguments
///
/// * `action` - The action to apply as an integer.
/// * `index` - The index of the game state to step.
#[ffi_function]
#[no_mangle]
pub extern "C" fn step_one(action: i32, index: i32) {
    let mut state = STATE.lock().unwrap();
    if let Some(game_state) = state.get_mut(index as usize) {
        if !game_state.is_done() {
            game_state.step_i32(action);
        }
    } else {
        panic!("Index {} out of bounds for batch size {}", index, state.len());
    }
}

/// Gets the observation for the game state at the specified index.
///
/// # Arguments
///
/// * `index` - The index of the game state to observe.
///
/// # Returns
///
/// * `observation::Observation` - The observation of the game state.
#[ffi_function]
#[no_mangle]
pub extern "C" fn get_one_observation(index: i32) -> Observation {
    let state = STATE.lock().unwrap();
    if let Some(game_state) = state.get(index as usize) {
        game_state.get_observation()
    } else {
        panic!("Index {} out of bounds for batch size {}", index, state.len());
    }
}

/// Closes the game state at the specified index.
/// The game state is reset and can be connected to again.
#[ffi_function]
#[no_mangle]
pub extern "C" fn close_one(index: i32) {
    let mut connected = CONNECTED.lock().unwrap();
    let mut state = STATE.lock().unwrap();
    if let Some(is_connected) = connected.get_mut(index as usize) {
        *is_connected = false;
        state.get_mut(index as usize).unwrap().reset();
    } else {
        panic!("Index {} out of bounds for batch size {}", index, connected.len());
    }
}

/// Gets the actions mask for the game state at the specified index.
/// The mask is an array of integers where 1 means the action will lead to something happening with the games state,
/// and 0 means taking the action will yield the same observation.
/// 
/// # Arguments
/// 
/// * `index` - The index of the game state to get the actions mask for.
/// 
/// # Returns
/// 
/// * `ActionMask` - The actions mask for the game state.
#[ffi_function]
#[no_mangle]
pub extern "C" fn valid_actions_mask(index: i32) -> ActionMask {
    let state = STATE.lock().unwrap();
    if let Some(game_state) = state.get(index as usize) {
        ActionMask::new(game_state)
    } else {
        panic!("Index {} out of bounds for batch size {}", index, state.len());
    }
}

/// Sets the record_replays setting to the given value.
/// Training is better done with record_replays set to false, as it saves memory and time.
/// For evaluation and assessment one can consider setting it to true.
/// Should be applied with a reset, as otherwise will produce incomplete replays for the currently running game states.
/// 
/// # Arguments
/// 
/// * `value` - The value to set record_replays to.
#[ffi_function]
#[no_mangle]
pub extern "C" fn set_record_replays(value: bool) {
    SETTINGS.write().unwrap().record_replays = value;
}

#[ffi_function]
#[no_mangle]
pub extern "C" fn num_actions() -> i32 {
    NUM_ACTIONS as i32
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
        .register(function!(set_batch_size))
        .register(function!(connect_env))
        .register(function!(reset))
        .register(function!(reset_one))
        .register(function!(step_one))
        .register(function!(get_one_observation))
        .register(function!(close_one))
        .register(function!(valid_actions_mask))
        .register(function!(set_record_replays))
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
fn verify_num_inventory_items() {
    use crate::observation::INVENTORY_SIZE;
    use game_logic::crafting::storable::ALL_STORABLES;
    println!("Actual length of the inventory: {}", ALL_STORABLES.len());
    assert_eq!(INVENTORY_SIZE, ALL_STORABLES.len(), "change INVENTORY_SIZE manually");
}

#[test]
fn verify_num_actions() {
    let mut max = 0;
    while let Ok(_) = Action::try_from(max) {
        max += 1;
    }
    println!("Actual number of actions: {}", max);
    assert_eq!(max, NUM_ACTIONS as i32, "change NUM_ACTIONS manually");
}
