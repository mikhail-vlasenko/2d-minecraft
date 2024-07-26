// use ffi::*;
// 
// fn main() {
//     println!("Initializing the stress test");
//     use rand::{Rng, thread_rng};
//     use std::time::{Duration, Instant};
// 
//     let max_action = 6;
//     let choose_action = || -> i32 {
//         thread_rng().gen_range(0..=max_action)
//     };
//     let batch = 32;
//     set_batch_size(batch);
//     set_record_replays(false);
// 
// 
//     let mut start = Instant::now();
//     let mut step_count = 0;
// 
//     println!("Starting the stress test");
//     loop {
//         for i in 0..batch {
//             step_one(choose_action(), i);
//             let _obs = get_one_observation(i);
//             step_count += 1;
//         }
// 
//         if start.elapsed() >= Duration::from_secs(1) {
//             println!("Steps run: {}", step_count);
//             step_count = 0;
//             // Reset the timer
//             start = Instant::now();
//         }
//     }
// }
fn main() {
    println!("Nothing")
}
