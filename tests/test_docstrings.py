"""The public API is documented by its docstrings, so every public name needs one."""

import inspect

import pytest

import spapprox


def _public_members(cls):
    """Public members of ``cls`` defined in spapprox, with the class defining them."""
    for name in dir(cls):
        if name.startswith("_"):
            continue
        owner = next(k for k in cls.__mro__ if name in k.__dict__)
        if owner.__module__.startswith("spapprox"):
            yield name, owner


def _api():
    for name in spapprox.__all__:
        obj = getattr(spapprox, name)
        yield name, obj
        if inspect.isclass(obj):
            for member, owner in _public_members(obj):
                raw = inspect.getattr_static(obj, member)
                # Plain data attributes (e.g. ``delta = None``) cannot carry a docstring
                if not isinstance(raw, (staticmethod, classmethod, property, type)) and not (
                    inspect.isfunction(raw)
                ):
                    continue
                yield f"{name}.{member} (defined in {owner.__name__})", raw


@pytest.mark.parametrize("name, obj", list(_api()), ids=lambda v: v if isinstance(v, str) else "")
def test_has_docstring(name, obj):
    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__
    assert inspect.getdoc(obj), f"{name} has no docstring"
