use lazy_static::lazy_static;
use std::sync::Mutex;
use std::boxed::Box;

const ENTRY_SIZE: usize = 128;
const NUM_ENTRIES: usize = 32;

lazy_static! {
    static ref EXECUTION_STATE_SPY: Mutex<[[u8; ENTRY_SIZE]; NUM_ENTRIES]> = Mutex::new([[0u8; ENTRY_SIZE]; NUM_ENTRIES]);
    static ref CURRENT_INDEX: Mutex<Box<usize>> = Mutex::new(Box::new(0));
}

/// This acts as near-0-overhead logging.
/// 
/// Call this function to be able to read it.
/// 
/// At an important place in the code, call the write_to_state_spy to "log" some short string into a fixed-size queue.
/// Using the python_wrapper/read_spy_data.py and plugging in the pid and the printed addresses from this function, 
/// it is possible to read the logging queue.
/// 
/// Why not just logging? 
/// 1. Because if the rust code hangs somewhere, the logging may not be flushed to disk, and you will be misled about where to look.
/// 2. Because this doesn't write anything to disk, so will not take up gigabytes when training an agent.
/// 
/// May not work well with multithreading.
pub fn locate_execution_state_spy() {
    let data_ptr = EXECUTION_STATE_SPY.lock().unwrap().as_ptr() as *const u8;
    let index_ptr = CURRENT_INDEX.lock().unwrap().as_ref() as *const usize;
    println!("Memory address at start: {:?}", data_ptr);
    println!("Index address: {:?}", index_ptr);
}

/// Push info to the fixed-size queue.
/// See locate_execution_state_spy for more info.
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
