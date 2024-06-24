import random

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size
from python_wrapper.observation import ProcessedObservation, get_processed_observation, get_action_name


init_lib('./target/debug/ffi.dll')

batch_size = 2
set_batch_size(batch_size)

num_actions = num_actions()
print(f'Total number of available actions: {num_actions}')

print(get_processed_observation(0))

step_one(0, 0)
print(get_processed_observation(0))

step_one(6, 0)
print(get_processed_observation(0))

for b in range(batch_size):
    for i in range(3):
        action = random.randint(0, num_actions - 1)
        print(f'Performing action: {get_action_name(action)}')
        step_one(action, b)
        print(get_processed_observation(b))

reset()
