from __future__ import annotations
import ctypes
import typing

T = typing.TypeVar("T")
c_lib = None

def init_lib(path):
    """Initializes the native library. Must be called at least once before anything else."""
    global c_lib
    c_lib = ctypes.cdll.LoadLibrary(path)

    c_lib.set_batch_size.argtypes = [ctypes.c_int32]
    c_lib.connect_env.argtypes = []
    c_lib.reset.argtypes = []
    c_lib.reset_one.argtypes = [ctypes.c_int32]
    c_lib.reset_one_to_saved.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int8)]
    c_lib.step_one.argtypes = [ctypes.c_int32, ctypes.c_int32]
    c_lib.get_one_observation.argtypes = [ctypes.c_int32]
    c_lib.close_one.argtypes = [ctypes.c_int32]
    c_lib.set_record_replays.argtypes = [ctypes.c_bool]
    c_lib.set_start_loadout.argtypes = [ctypes.c_int32]
    c_lib.set_save_on_milestone.argtypes = [ctypes.c_bool]
    c_lib.get_batch_size.argtypes = []
    c_lib.action_name.argtypes = [ctypes.c_int32]

    c_lib.connect_env.restype = ctypes.c_int32
    c_lib.get_one_observation.restype = Observation
    c_lib.get_batch_size.restype = ctypes.c_int32
    c_lib.action_name.restype = ctypes.POINTER(ctypes.c_int8)



def set_batch_size(new_batch_size: int):
    """ Initializes batch_size games states.
 Does a reset on all game states."""
    return c_lib.set_batch_size(new_batch_size)

def connect_env() -> int:
    """ Finds an unconnected games state and returns its index.
 Panics if all game states are connected."""
    return c_lib.connect_env()

def reset():
    """ Resets all game states.
 Not advised. Especially since game states initialize ready to be stepped."""
    return c_lib.reset()

def reset_one(index: int):
    """ Resets the game state at the specified index."""
    return c_lib.reset_one(index)

def reset_one_to_saved(index: int, save_name: ctypes.POINTER(ctypes.c_int8)):
    """ Sets the state of this game to the one from a save file."""
    return c_lib.reset_one_to_saved(index, save_name)

def step_one(action: int, index: int):
    """ Steps the game state at the specified index with the given action.

 # Arguments

 * `action` - The action to apply as an integer.
 * `index` - The index of the game state to step."""
    return c_lib.step_one(action, index)

def get_one_observation(index: int) -> Observation:
    """ Gets the observation for the game state at the specified index.

 # Arguments

 * `index` - The index of the game state to observe.

 # Returns

 * `observation::Observation` - The observation of the game state."""
    return c_lib.get_one_observation(index)

def close_one(index: int):
    """ Closes the game state at the specified index.
 The game state is reset and can be connected to again."""
    return c_lib.close_one(index)

def set_record_replays(value: bool):
    """ Sets the record_replays setting to the given value.
 Training is better done with record_replays set to false, as it saves memory and time.
 For evaluation and assessment one can consider setting it to true.
 Should be applied with a reset, as otherwise will produce incomplete replays for the currently running game states.
 
 # Arguments
 
 * `value` - The value to set record_replays to."""
    return c_lib.set_record_replays(value)

def set_start_loadout(value_index: int):
    """ Sets the start loadout for the player for all new games.
 
 # Arguments
 
 * `value_index` - The index of the loadout to set. Not the actual string value because of the FFI memory safety."""
    return c_lib.set_start_loadout(value_index)

def set_save_on_milestone(value: bool):
    """ Sets the save_on_milestone setting to the given value.
 
 # Arguments
 
 * `value` - The value to set save_on_milestone to."""
    return c_lib.set_save_on_milestone(value)

def get_batch_size() -> int:
    return c_lib.get_batch_size()

def action_name(action: int) -> ctypes.POINTER(ctypes.c_int8):
    return c_lib.action_name(action)



OBSERVATION_GRID_SIZE = 17
INVENTORY_SIZE = 26
NUM_ACTIONS = 39
MOB_INFO_SIZE = 8
MAX_MOBS = 16
LOOT_INFO_SIZE = 3
NUM_MATERIALS = 13


TRUE = ctypes.c_uint8(1)
FALSE = ctypes.c_uint8(0)


def _errcheck(returned, success):
    """Checks for FFIErrors and converts them to an exception."""
    if returned == success: return
    else: raise Exception(f"Function returned error: {returned}")


class CallbackVars(object):
    """Helper to be used `lambda x: setattr(cv, "x", x)` when getting values from callbacks."""
    def __str__(self):
        rval = ""
        for var in  filter(lambda x: "__" not in x, dir(self)):
            rval += f"{var}: {getattr(self, var)}"
        return rval


class _Iter(object):
    """Helper for slice iterators."""
    def __init__(self, target):
        self.i = 0
        self.target = target

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.target.len:
            raise StopIteration()
        rval = self.target[self.i]
        self.i += 1
        return rval


class Observation(ctypes.Structure):

    # These fields represent the underlying C data layout
    _fields_ = [
        ("top_materials", ctypes.c_int32 * 17 * 17),
        ("tile_heights", ctypes.c_int32 * 17 * 17),
        ("player_pos", ctypes.c_int32 * 3),
        ("player_rot", ctypes.c_int32),
        ("hp", ctypes.c_int32),
        ("time", ctypes.c_float),
        ("inventory_state", ctypes.c_int32 * 26),
        ("mobs", ctypes.c_int32 * 8 * 16),
        ("loot", ctypes.c_int32 * 3 * 16),
        ("action_mask", ctypes.c_int32 * 39),
        ("score", ctypes.c_int32),
        ("message", ctypes.POINTER(ctypes.c_int8)),
        ("done", ctypes.c_bool),
    ]

    def __init__(self, top_materials = None, tile_heights = None, player_pos = None, player_rot: int = None, hp: int = None, time: float = None, inventory_state = None, mobs = None, loot = None, action_mask = None, score: int = None, message: ctypes.POINTER(ctypes.c_int8) = None, done: bool = None):
        if top_materials is not None:
            self.top_materials = top_materials
        if tile_heights is not None:
            self.tile_heights = tile_heights
        if player_pos is not None:
            self.player_pos = player_pos
        if player_rot is not None:
            self.player_rot = player_rot
        if hp is not None:
            self.hp = hp
        if time is not None:
            self.time = time
        if inventory_state is not None:
            self.inventory_state = inventory_state
        if mobs is not None:
            self.mobs = mobs
        if loot is not None:
            self.loot = loot
        if action_mask is not None:
            self.action_mask = action_mask
        if score is not None:
            self.score = score
        if message is not None:
            self.message = message
        if done is not None:
            self.done = done

    @property
    def top_materials(self):
        return ctypes.Structure.__get__(self, "top_materials")

    @top_materials.setter
    def top_materials(self, value):
        return ctypes.Structure.__set__(self, "top_materials", value)

    @property
    def tile_heights(self):
        return ctypes.Structure.__get__(self, "tile_heights")

    @tile_heights.setter
    def tile_heights(self, value):
        return ctypes.Structure.__set__(self, "tile_heights", value)

    @property
    def player_pos(self):
        return ctypes.Structure.__get__(self, "player_pos")

    @player_pos.setter
    def player_pos(self, value):
        return ctypes.Structure.__set__(self, "player_pos", value)

    @property
    def player_rot(self) -> int:
        return ctypes.Structure.__get__(self, "player_rot")

    @player_rot.setter
    def player_rot(self, value: int):
        return ctypes.Structure.__set__(self, "player_rot", value)

    @property
    def hp(self) -> int:
        return ctypes.Structure.__get__(self, "hp")

    @hp.setter
    def hp(self, value: int):
        return ctypes.Structure.__set__(self, "hp", value)

    @property
    def time(self) -> float:
        return ctypes.Structure.__get__(self, "time")

    @time.setter
    def time(self, value: float):
        return ctypes.Structure.__set__(self, "time", value)

    @property
    def inventory_state(self):
        return ctypes.Structure.__get__(self, "inventory_state")

    @inventory_state.setter
    def inventory_state(self, value):
        return ctypes.Structure.__set__(self, "inventory_state", value)

    @property
    def mobs(self):
        return ctypes.Structure.__get__(self, "mobs")

    @mobs.setter
    def mobs(self, value):
        return ctypes.Structure.__set__(self, "mobs", value)

    @property
    def loot(self):
        return ctypes.Structure.__get__(self, "loot")

    @loot.setter
    def loot(self, value):
        return ctypes.Structure.__set__(self, "loot", value)

    @property
    def action_mask(self):
        return ctypes.Structure.__get__(self, "action_mask")

    @action_mask.setter
    def action_mask(self, value):
        return ctypes.Structure.__set__(self, "action_mask", value)

    @property
    def score(self) -> int:
        return ctypes.Structure.__get__(self, "score")

    @score.setter
    def score(self, value: int):
        return ctypes.Structure.__set__(self, "score", value)

    @property
    def message(self) -> ctypes.POINTER(ctypes.c_int8):
        return ctypes.Structure.__get__(self, "message")

    @message.setter
    def message(self, value: ctypes.POINTER(ctypes.c_int8)):
        return ctypes.Structure.__set__(self, "message", value)

    @property
    def done(self) -> bool:
        return ctypes.Structure.__get__(self, "done")

    @done.setter
    def done(self, value: bool):
        return ctypes.Structure.__set__(self, "done", value)




class callbacks:
    """Helpers to define callbacks."""


