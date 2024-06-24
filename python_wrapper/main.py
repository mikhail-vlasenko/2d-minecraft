import ctypes
import numpy as np
import random

from python_wrapper.ffi_elements import init_lib, reset, step, num_actions
from python_wrapper.observation import ProcessedObservation, get_processed_observation, get_action_name


init_lib('./target/debug/ffi.dll')


num_actions = num_actions()
print(f'Total number of available actions: {num_actions}')

reset()
print(get_processed_observation())

step(0)
print(get_processed_observation())

step(6)
print(get_processed_observation())

for i in range(3):
    action = random.randint(0, num_actions - 1)
    print(f'Performing action: {get_action_name(action)}')
    step(action)
    print(get_processed_observation())
