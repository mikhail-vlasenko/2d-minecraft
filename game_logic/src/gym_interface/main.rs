use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;


#[no_mangle]
pub extern "C" fn hello_from_rust() {
    println!("Hello from Rust!");
}

#[no_mangle]
pub extern "C" fn reset(seed: i32, options: *mut *mut c_char, options_len: usize) -> *mut i32 {
    // Create an array on the heap
    let result = Box::new([1, 2, 3]);
    let result_ptr = Box::into_raw(result);

    // Example of processing options
    if !options.is_null() {
        let options_slice = unsafe { std::slice::from_raw_parts(options, options_len) };
        for &option_ptr in options_slice {
            let option_cstr = unsafe { CString::from_raw(option_ptr) };
            println!("Option: {:?}", option_cstr);
            // Do something with the option string
            // Note: CString::from_raw() takes ownership, so the memory will be freed when the CString is dropped
        }
    }

    result_ptr as *mut i32
}
