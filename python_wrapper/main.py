import random

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size
from python_wrapper.observation import ProcessedObservation, get_processed_observation, get_action_name, \
    get_actions_mask

init_lib('./target/release/ffi.dll')

batch_size = 2
set_batch_size(batch_size)

num_actions = num_actions()
print(f'Total number of available actions: {num_actions}')
for i in range(num_actions):
    print(f'Action {i}: {get_action_name(i)}')

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

        mask = get_actions_mask(b)
        print(f'Valid actions for game state {b}: {mask}')

reset()
