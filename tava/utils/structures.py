# Copyright (c) Meta Platforms, Inc. and affiliates.
import collections

Rays = collections.namedtuple(
    "Rays", ("origins", "directions", "viewdirs", "radii", "near", "far")
)
Bones = collections.namedtuple("Bones", ("heads", "tails", "transforms"))
Cameras = collections.namedtuple(
    "Cameras", ("intrins", "extrins", "distorts", "width", "height")
)


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))
