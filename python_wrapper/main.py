import ctypes
import numpy as np

from python_wrapper.observation import CObservation, Observation

# Load the shared library
lib = ctypes.CDLL('./target/debug/ffi.dll')  # Replace with the actual path to your DLL

# Define function prototypes
lib.hello_from_rust.restype = None
lib.reset.restype = None
lib.get_observation.restype = CObservation

# Call the Rust function `hello_from_rust`
lib.hello_from_rust()

# Call the Rust function `reset`
lib.reset()

# Call the Rust function `get_observation` and get the result
c_observation = lib.get_observation()
observation = Observation.from_c_observation(c_observation)
print(observation)

lib.step(0)
c_observation = lib.get_observation()
observation = Observation.from_c_observation(c_observation)
print(observation)

lib.step(6)
c_observation = lib.get_observation()
observation = Observation.from_c_observation(c_observation)
print(observation)

