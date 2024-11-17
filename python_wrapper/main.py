import random

from python_wrapper.ffi_elements import init_lib, reset, step_one, num_actions, set_batch_size, connect_env
from python_wrapper.observation import ProcessedObservation, get_processed_observation, get_action_name, \
    get_actions_mask, reset_one_to_saved_wrapped

init_lib('./target/release/ffi.dll')

batch_size = 2
set_batch_size(batch_size)

num_actions = num_actions()
print(f'Total number of available actions: {num_actions}')
for i in range(num_actions):
    print(f'Action {i}: {get_action_name(i)}')

print(f'first connection index: {connect_env()}')
print(f'second connection index: {connect_env()}')

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

test_save_name = 'autosave_ms_1_score_97_2024-11-17_18-30-46'
reset_one_to_saved_wrapped(0, test_save_name)
print(get_processed_observation(0))

reset()
