import os
import pickle
import tempfile
import cloudpickle
import joblib
import logging
import sys
import types
logger = logging.getLogger(__name__)

try:
    from cedarkit.utils.cli import setup_logging, log_line
    from cedarkit.core.data_objects import DataGroup, OutputCollection, RunConfig, Output
    from cedarkit.viz.grids import GridCell
    from cedarkit.core.relationship import Relationship
except ImportError:
    # Fallback: imports when running as a package
    from utils.cli.logging import setup_logging, log_line
    from core.data_objects import OutputCollection, RunConfig, Output
    from viz.grids import GridCell
    from core.relationship import Relationship

# import your *current* classes




def _ensure_module(modname: str) -> types.ModuleType:
    """Create or return a module object with this name, wiring up parents if needed."""
    if modname in sys.modules:
        return sys.modules[modname]

    if "." in modname:
        parent_name, _, leaf = modname.rpartition(".")
        parent_mod = _ensure_module(parent_name)
        mod = types.ModuleType(modname)
        setattr(parent_mod, leaf, mod)
    else:
        mod = types.ModuleType(modname)

    sys.modules[modname] = mod
    return mod


def _install_unpickling_aliases():
    """
    Backward-compatible paths observed in old pickles:

      data_obj.plotting_objects.GridCell
      data_obj.data_objects.OutputCollection
      data_obj.data_objects.DataGroup
      data_obj.data_objects.RunConfig
      data_obj.data_objects.Output
      data_obj.relationship_obj.Relationship
      grp_config.RunConfig
      table.Output
    """

    aliases = {
        ("data_obj.plotting_objects", "GridCell"): GridCell,
        ("data_obj.data_objects", "OutputCollection"): OutputCollection,
        ("data_obj.data_objects", "DataGroup"): DataGroup,
        ("data_obj.data_objects", "RunConfig"): RunConfig,
        ("data_obj.data_objects", "Output"): Output,          # <-- new
        ("data_obj.relationship_obj", "Relationship"): Relationship,
        ("grp_config", "RunConfig"): RunConfig,
        ("table", "Output"): Output,
    }

    for (modname, attr), cls in aliases.items():
        mod = _ensure_module(modname)
        setattr(mod, attr, cls)

def joblib_cloud_load(path):
    _install_unpickling_aliases()     # must be before loads()
    blob = joblib.load(path)          # bytes from cloudpickle.dumps(obj)
    return cloudpickle.loads(blob)


def _atomic_write(path, writer):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        try: os.remove(tmp)
        except OSError: pass


def joblib_cloud_atomic_dump(obj, path, *, compress=3, protocol=pickle.HIGHEST_PROTOCOL):
    blob = cloudpickle.dumps(obj, protocol=protocol)
    _atomic_write(path, lambda tmp: joblib.dump(blob, tmp, compress=compress))


# def joblib_cloud_load(path):
#     blob = joblib.load(path)
#     return cloudpickle.loads(blob)

# in cedarkit/utils/io/cloudjoblib.py

# def joblib_cloud_load(path):
    # import sys, types
    # import joblib, cloudpickle
    # import cedarkit.core.data_objects as new_data_obj  # where DataGroup lives now
    # import cedarkit.viz.grids as new_grid_obj  # where DataGroup lives now
    #
    # # Fake package 'data_obj'
    # if "data_obj" not in sys.modules:
    #     pkg = types.ModuleType("data_obj")
    #     pkg.__path__ = []  # mark as package-like
    #     sys.modules["data_obj"] = pkg
    #
    # # Fake module 'data_obj.data_objects' that exposes DataGroup
    # if "data_obj.data_objects" not in sys.modules:
    #     old_mod = types.ModuleType("data_obj.data_objects")
    #     old_mod.OutputCollection = new_data_obj.OutputCollection   # crucial line
    #     sys.modules["data_obj.data_objects"] = old_mod
    #     sys.modules["data_obj"].data_objects = old_mod
    # if "data_obj.plotting_objects" not in sys.modules:
    #     old_mod = types.ModuleType("data_obj.plotting_objects")
    #     old_mod.GridCell = new_grid_obj.GridCell   # crucial line
    #     sys.modules["data_obj.plotting_objects"] = old_mod
    #     sys.modules["data_obj"].plotting_objects = old_mod
    #

    # blob = joblib.load(path)
    # return cloudpickle.loads(blob)



def joblib_atomic_dump(obj, path, *, compress=3, protocol=None):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    os.close(fd)
    try:
        joblib.dump(obj, tmp, compress=compress, protocol=protocol)
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass


def joblib_safe_load(path, *, mmap_mode=None):
    # Try a strict load first; if it fails with EOF, surface a clear message.
    try:
        return joblib.load(path, mmap_mode=mmap_mode)
    except EOFError as e:
        raise EOFError(f"{path} appears truncated/corrupted. "
                       "Recreate it with an atomic dump and avoid concurrent writers.") from e
