from __future__ import annotations
import ctypes
import typing

T = typing.TypeVar("T")
c_lib = None

def init_lib(path):
    """Initializes the native library. Must be called at least once before anything else."""
    global c_lib
    c_lib = ctypes.cdll.LoadLibrary(path)

    c_lib.reset.argtypes = []
    c_lib.step.argtypes = [ctypes.c_int32]
    c_lib.get_observation.argtypes = []

    c_lib.get_observation.restype = Observation



def reset():
    return c_lib.reset()

def step(action: int):
    return c_lib.step(action)

def get_observation() -> Observation:
    return c_lib.get_observation()





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
        ("inventory_state", ctypes.c_int32 * 128),
        ("mobs", ctypes.c_int32 * 4 * 16),
        ("message", ctypes.POINTER(ctypes.c_int8)),
    ]

    def __init__(self, top_materials = None, tile_heights = None, player_pos = None, player_rot: int = None, hp: int = None, time: float = None, inventory_state = None, mobs = None, message: ctypes.POINTER(ctypes.c_int8) = None):
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
        if message is not None:
            self.message = message

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
    def message(self) -> ctypes.POINTER(ctypes.c_int8):
        return ctypes.Structure.__get__(self, "message")

    @message.setter
    def message(self, value: ctypes.POINTER(ctypes.c_int8)):
        return ctypes.Structure.__set__(self, "message", value)




class callbacks:
    """Helpers to define callbacks."""


