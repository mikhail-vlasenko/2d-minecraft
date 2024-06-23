import ctypes
import numpy as np

from python_wrapper.observation import Observation, OBSERVATION_GRID_SIZE

# Load the shared library
lib = ctypes.CDLL('./target/debug/ffi.dll')  # Replace with the actual path to your DLL

# Define function prototypes
lib.hello_from_rust.restype = None
lib.reset.restype = None
lib.get_observation.restype = Observation

# Call the Rust function `hello_from_rust`
lib.hello_from_rust()

# Call the Rust function `reset`
lib.reset()

# Call the Rust function `get_observation` and get the result
observation = lib.get_observation()

# Convert the Observation structure to NumPy arrays for easier manipulation
top_materials = np.ctypeslib.as_array(observation.top_materials).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))
tile_depths = np.ctypeslib.as_array(observation.tile_depths).reshape((OBSERVATION_GRID_SIZE, OBSERVATION_GRID_SIZE))

# Print the results
print("Top Materials:")
print(top_materials)

print("Tile Depths:")
print(tile_depths)
