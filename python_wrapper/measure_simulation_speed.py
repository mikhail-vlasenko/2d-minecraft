import time
from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size, set_record_replays


def measure_ticks_per_second(duration_seconds=3):
    batch_size = 32
    set_batch_size(batch_size)
    set_record_replays(False)

    start_time = time.time()
    end_time = start_time + duration_seconds
    tick_count = 0

    while time.time() < end_time:
        for b in range(batch_size):
            # go up
            step_one(0, b)
            # go down
            step_one(2, b)
        tick_count += 2 * batch_size

    elapsed_time = time.time() - start_time
    ticks_per_second = tick_count / elapsed_time
    return ticks_per_second


if __name__ == '__main__':
    # Ticks per second: 2267.8793700761503
    # init_lib('./target/debug/ffi.dll')
    # Ticks per second: 30087.898636752117
    init_lib('./target/release/ffi.dll')
    ticks_per_second = measure_ticks_per_second()
    print(f'Ticks per second: {ticks_per_second}')
