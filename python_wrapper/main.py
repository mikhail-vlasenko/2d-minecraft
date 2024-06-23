import ctypes
import numpy as np

from python_wrapper.ffi_elements import init_lib, Observation, reset, step, get_observation
from python_wrapper.observation import ProcessedObservation


lib = init_lib('./target/debug/ffi.dll')


reset()

# Call the Rust function `get_observation` and get the result
c_observation = get_observation()
observation = ProcessedObservation.from_c_observation(c_observation)
print(observation)

step(0)
c_observation = get_observation()
observation = ProcessedObservation.from_c_observation(c_observation)
print(observation)

step(6)
c_observation = get_observation()
observation = ProcessedObservation.from_c_observation(c_observation)
print(observation)

step(20)
c_observation = get_observation()
observation = ProcessedObservation.from_c_observation(c_observation)
print(observation)
