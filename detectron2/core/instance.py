import importlib.util
import os
import sys
import tempfile
import torch

import detectron2
from detectron2.structures import Boxes  # noqa F401

_fields = {}
_counter = 0


@torch.jit.script
class Instances:
    pass


def _gen_class():
    def indent(level, s):
        return " " * 4 * level + s

    lines = []

    cls_name = "Instance{}".format(_counter)
    lines.append("class {}:".format(cls_name))
    lines.append("")

    lines.append(indent(1, "def __init__(self, image_size: Tuple[int, int]):"))
    lines.append(indent(2, "self.image_size = image_size"))
    for name, type_ in _fields.items():
        lines.append(
            indent(2, "self.{} = torch.jit.annotate(Optional[{}], None)".format(name, type_))
        )
    lines.append("")

    # for name, type_ in _fields.items():
    #     # getter
    #     lines.append(indent(1, "def get_{}(self) -> {}:".format(name, type_)))
    #     lines.append(indent(2, "val = self.{}".format(name)))
    #     lines.append(indent(2, "assert val is not None"))
    #     lines.append(indent(2, "return val"))
    #     lines.append("")
    #     # setter
    #     lines.append(indent(1, "def set_{}(self, val: {}):".format(name, type_)))
    #     lines.append(indent(2, "self.{} = val".format(name)))
    #     lines.append("")

    return cls_name, os.linesep.join(lines)


def _typing_imports_str():
    lines = []
    lines.append("import torch")
    lines.append("from torch import Tensor")
    lines.append("import typing")
    lines.append("from typing import *")
    lines.append("")
    lines.append("")
    lines.append("from detectron2.structures import Boxes")
    lines.append("")
    lines.append("")
    return os.linesep.join(lines)


def _import(path):
    spec = importlib.util.spec_from_file_location(
        "{}{}".format(sys.modules[__name__].__name__, _counter), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


def _rescript():
    global Instances
    qualified_name = torch._jit_internal._qualified_name(Instances)

    global _counter
    _counter += 1

    cls_name, cls_def = _gen_class()
    typing_imports = _typing_imports_str()

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".py", delete=False) as f:
        f.write(typing_imports)
        f.write(cls_def)
        f.flush()
        module = _import(f.name)
        new_instance = torch.jit.script(getattr(module, cls_name))
        Instances = new_instance
        torch.jit._add_script_class(new_instance, qualified_name)
        detectron2.structures.Instances.__torch_script_class__ = True
        detectron2.structures.Instances._jit_override_qualname = torch.jit._qualified_name(new_instance)


def register_fields(fields):
    changed = False
    for name, type_ in fields.items():
        if name not in _fields:
            _fields[name] = type_
            changed = True
        else:
            if _fields[name] != type_:
                raise ValueError(
                    "Field {} has already been registered as type {} before!".format(name, type_)
                )
    if changed:
        _rescript()
