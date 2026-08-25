import os
import pickle
import tempfile
import cloudpickle
import joblib
import logging
import sys
import types
import pathlib
logger = logging.getLogger(__name__)

from cedarkit.utils.cli import setup_logging, log_line
from cedarkit.core.data_objects import DataGroup, OutputCollection, RunConfig, Output
from cedarkit.viz.grids import GridCell
from cedarkit.core.relationship import Relationship

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
    """Load a joblib file whose contents were serialized with cloudpickle.

    Registers backward-compatible module aliases (see
    ``_install_unpickling_aliases``) before unpickling, so joblib files
    created by older CedarKit versions — whose classes lived at paths like
    ``data_obj.data_objects.DataGroup`` rather than their current
    ``cedarkit.core.data_objects.DataGroup`` — still unpickle correctly.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a joblib file previously written by
        ``joblib_cloud_atomic_dump`` (or the older, non-atomic
        ``cloudpickle.dumps`` + ``joblib.dump`` pattern it replaced).

    Returns
    -------
    object
        The unpickled object.

    See Also
    --------
    joblib_cloud_atomic_dump : Writes files in the format this function reads.
    """
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
    """Serialize an object with cloudpickle and write it atomically via joblib.

    Cloudpickle handles objects joblib/pickle alone can't (e.g. objects
    defined in a notebook's ``__main__``, or closures); the write itself
    goes to a temporary file in the same directory and is moved into place
    with ``os.replace`` (atomic on POSIX), so a crash mid-write can't leave
    ``path`` truncated.

    Parameters
    ----------
    obj : object
        Object to serialize. Anything ``cloudpickle.dumps`` can handle.
    path : str or pathlib.Path
        Destination file path.
    compress : int, default 3
        Compression level passed to ``joblib.dump``.
    protocol : int, default pickle.HIGHEST_PROTOCOL
        Pickle protocol passed to ``cloudpickle.dumps``.

    See Also
    --------
    joblib_cloud_load : Reads files written by this function.
    """
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
