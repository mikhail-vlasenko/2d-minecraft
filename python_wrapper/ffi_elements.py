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
    c_lib.reset.argtypes = []
    c_lib.step_one.argtypes = [ctypes.c_int32, ctypes.c_int32]
    c_lib.get_one_observation.argtypes = [ctypes.c_int32]
    c_lib.valid_actions_mask.argtypes = [ctypes.c_int32]
    c_lib.set_record_replays.argtypes = [ctypes.c_bool]
    c_lib.num_actions.argtypes = []
    c_lib.action_name.argtypes = [ctypes.c_int32]

    c_lib.get_one_observation.restype = Observation
    c_lib.valid_actions_mask.restype = ActionMask
    c_lib.num_actions.restype = ctypes.c_int32
    c_lib.action_name.restype = ctypes.POINTER(ctypes.c_int8)



def set_batch_size(new_batch_size: int):
    """ Does a reset on all game states"""
    return c_lib.set_batch_size(new_batch_size)

def reset():
    """ Resets all game states.
 Not advised. Especially since game states initialize ready to be stepped."""
    return c_lib.reset()

def step_one(action: int, index: int):
    """ Steps the game state at the specified index with the given action.
 Checking for Done is at the start of the function. 
 An action that was sent to a game that was already done, will be ignored, 
 so that a starting observation can be obtained. 

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

def valid_actions_mask(index: int) -> ActionMask:
    """ Gets the actions mask for the game state at the specified index.
 The mask is an array of integers where 1 means the action will lead to something happening with the games state,
 and 0 means taking the action will yield the same observation.
 
 # Arguments
 
 * `index` - The index of the game state to get the actions mask for.
 
 # Returns
 
 * `ActionMask` - The actions mask for the game state."""
    return c_lib.valid_actions_mask(index)

def set_record_replays(value: bool):
    """ Sets the record_replays setting to the given value.
 Training is better done with record_replays set to false, as it saves memory and time.
 For evaluation and assessment one can consider setting it to true.
 Should be applied with a reset, as otherwise will produce incomplete replays for the currently running game states.
 
 # Arguments
 
 * `value` - The value to set record_replays to."""
    return c_lib.set_record_replays(value)

def num_actions() -> int:
    return c_lib.num_actions()

def action_name(action: int) -> ctypes.POINTER(ctypes.c_int8):
    return c_lib.action_name(action)





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


class ActionMask(ctypes.Structure):

    # These fields represent the underlying C data layout
    _fields_ = [
        ("mask", ctypes.c_int32 * 39),
    ]

    def __init__(self, mask = None):
        if mask is not None:
            self.mask = mask

    @property
    def mask(self):
        return ctypes.Structure.__get__(self, "mask")

    @mask.setter
    def mask(self, value):
        return ctypes.Structure.__set__(self, "mask", value)


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
        ("mobs", ctypes.c_int32 * 4 * 16),
        ("score", ctypes.c_int32),
        ("message", ctypes.POINTER(ctypes.c_int8)),
        ("done", ctypes.c_bool),
    ]

    def __init__(self, top_materials = None, tile_heights = None, player_pos = None, player_rot: int = None, hp: int = None, time: float = None, inventory_state = None, mobs = None, score: int = None, message: ctypes.POINTER(ctypes.c_int8) = None, done: bool = None):
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


