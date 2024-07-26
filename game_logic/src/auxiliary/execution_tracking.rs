use lazy_static::lazy_static;
use std::sync::Mutex;
use std::boxed::Box;

const ENTRY_SIZE: usize = 128;
const NUM_ENTRIES: usize = 16;

lazy_static! {
    static ref EXECUTION_STATE_SPY: Mutex<[[u8; ENTRY_SIZE]; NUM_ENTRIES]> = Mutex::new([[0u8; ENTRY_SIZE]; NUM_ENTRIES]);
    static ref CURRENT_INDEX: Mutex<Box<usize>> = Mutex::new(Box::new(0));
}

/// This may not work well with multithreading.
pub fn init_execution_state_spy() {
    let data_ptr = EXECUTION_STATE_SPY.lock().unwrap().as_ptr() as *const u8;
    let index_ptr = CURRENT_INDEX.lock().unwrap().as_ref() as *const usize;
    println!("Memory address at start: {:?}", data_ptr);
    println!("Index address: {:?}", index_ptr);
}

pub fn write_to_state_spy(input: &str) {
    let mut state_spy = EXECUTION_STATE_SPY.lock().unwrap();
    let mut current_index = CURRENT_INDEX.lock().unwrap();

    let bytes = input.as_bytes();
    let length = bytes.len();

    // Write the input to the current index
    state_spy[**current_index][..length].copy_from_slice(bytes);

    // Zero out the rest of the buffer to avoid leftover data
    for i in length..ENTRY_SIZE {
        state_spy[**current_index][i] = 0;
    }

    // Update the index to point to the next entry
    **current_index = (**current_index + 1) % NUM_ENTRIES;
}
