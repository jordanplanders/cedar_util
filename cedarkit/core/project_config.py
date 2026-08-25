from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Mapping

import yaml
from pathlib import Path

import cedarkit.utils.routing.paths


class ProjectConfig:
    """Recursive attribute-style wrapper around a nested config dict.

    Any dict value (including nested dicts of dicts) becomes a nested
    ``ProjectConfig`` whose keys are exposed as attributes, so config values
    loaded from YAML/JSON can be accessed as ``config.section.key`` instead
    of ``config['section']['key']``. There is no fixed schema — attribute
    names are whatever keys were present in the source data.

    ``to_dict()``/``save_config()`` provide the inverse direction (back to a
    plain dict / YAML file). ``get_dynamic_attr`` supports looking up an
    attribute chain where one segment is a runtime variable rather than a
    literal name. ``set_entry``/``set_consolidated_format`` (and their
    getters) are runtime-only flags not persisted by ``to_dict``/``save_config``.
    """

    def __init__(self, config_data, file_path=None, is_root=True):
        """Build a ``ProjectConfig`` by recursively wrapping ``config_data``.

        Parameters
        ----------
        config_data : dict
            Source mapping. Any value that is itself a ``dict`` becomes a
            nested ``ProjectConfig`` (with ``is_root=False``); other values
            are set directly as attributes. A ``"file_path"`` key in
            ``config_data`` is ignored (not set as an attribute) to avoid
            duplicating/overwriting the ``file_path`` set below.
        file_path : str or pathlib.Path, optional
            Path this config was loaded from. Only stored (as ``self.file_path``)
            when ``is_root`` is ``True`` — nested ``ProjectConfig`` instances
            don't carry a ``file_path``.
        is_root : bool, optional
            Whether this is the top-level instance (vs. a nested one created
            recursively for a dict value). Default is ``True``.
        """
        if is_root:
            self.file_path = str(file_path) if file_path else None

        for key, value in config_data.items():
            if key == "file_path":
                continue  # don't overwrite or duplicate this manually
            if isinstance(value, dict):
                setattr(self, key, ProjectConfig(value, is_root=False))
            else:
                setattr(self, key, value)

    def __repr__(self):
        # Returns a string showing the class name and full __dict__.
        return f"{self.__class__.__name__}({self.__dict__})"

    # Canonical home for project-directory resolution previously duplicated in
    # graphccm.utils.time_axis and graphccm.utils.sampling.meta_master.
    @property
    def project_dir(self) -> Path:
        """Directory containing the source project configuration file."""
        if self.file_path is None:
            raise ValueError("ProjectConfig must have file_path set to resolve project_dir.")
        return Path(self.file_path).resolve().parent

    def has_nested_attribute(self, attr_chain):
        """Check whether a dotted attribute chain resolves on this instance.

        Parameters
        ----------
        attr_chain : str
            Dot-separated attribute path, e.g. ``"a.b.c"``.

        Returns
        -------
        bool
            ``True`` if every segment of the chain exists, ``False`` as soon
            as one is missing.
        """
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    def add_attribute(self, key, value):
        # Mutator: sets self.<key> to value (wrapped in a nested ProjectConfig if value is a dict). No return value.
        if isinstance(value, dict):
            setattr(self, key, ProjectConfig(value))
        else:
            setattr(self, key, value)

    def add_to_list(self, list_name, item):
        # Mutator: appends item to self.<list_name> in place. No return value.
        # Raises TypeError if self.<list_name> isn't already a list (including if it's unset).
        current_list = getattr(self, list_name, None)
        if isinstance(current_list, list):
            current_list.append(item)
        else:
            raise TypeError(f"{list_name} is not a list.")

    # Routing helpers (runtime-only flags; not persisted to YAML)
    def set_entry(self, entry: str):
        """Set entry mode for routing (e.g., 'sqlite' or 'csv')."""
        self._entry = str(entry) if entry is not None else None

    def get_entry(self):
        # Returns self._entry, or None if it was never set via set_entry.
        return getattr(self, "_entry", None)

    def set_consolidated_format(self, fmt: str):
        """Override consolidated output format (default: parquet for csv entry)."""
        self._consolidated_format = str(fmt) if fmt is not None else None

    def get_consolidated_format(self):
        # Returns self._consolidated_format, or None if it was never set via set_consolidated_format.
        return getattr(self, "_consolidated_format", None)

    def to_dict(self):
        """Recursively convert this config back into a plain dict.

        Nested ``ProjectConfig`` attributes are converted via their own
        ``to_dict()``. The keys ``"file_path"`` and ``"_data_vars_loaded"``
        are always excluded, so round-tripping through ``to_dict()`` is
        lossy by design for those two — ``file_path`` because it's
        re-derived rather than stored in the dict form, and
        ``_data_vars_loaded`` because it's bookkeeping from
        `load_config`, not config data.

        Returns
        -------
        dict
            Plain nested dict representation of this config.
        """
        result = {}
        for key, value in self.__dict__.items():
            if key in {"file_path", "_data_vars_loaded"}:
                continue
            if isinstance(value, ProjectConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def save_config(self):
        """Write this config's ``to_dict()`` representation back to YAML.

        Only works when called on the root instance, since only the root
        has ``self.file_path`` set (see `__init__`).

        Raises
        ------
        ValueError
            If ``self.file_path`` is ``None``.
        """
        if self.file_path is None:
            raise ValueError("No file path specified for saving the configuration.")
        with open(self.file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    # Replaces graphccm.utils.time_axis._persist_yaml_updates.
    def patch_source(self, updates: Mapping[str, Any]) -> None:
        """Patch dotted keys in the source YAML without serializing merged config.

        ``load_config`` expands external variable configuration into this
        object. This method rereads and patches the source YAML so those
        expanded runtime fields are not written back into the project file.
        The completed YAML is atomically substituted for the original file.
        """
        if self.file_path is None:
            raise ValueError("ProjectConfig must have file_path set to patch its source.")

        source_path = Path(self.file_path).resolve()
        raw = _load_yaml(source_path)
        for key_path, value in updates.items():
            parts = str(key_path).split(".")
            if not all(parts):
                raise ValueError(f"Invalid configuration key path: {key_path!r}")
            current = raw
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

        original_mode = source_path.stat().st_mode & 0o7777
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=source_path.parent,
                prefix=f".{source_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                yaml.safe_dump(raw, temporary_file, sort_keys=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            temporary_path.chmod(original_mode)
            os.replace(temporary_path, source_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_dynamic_attr(self, attr_chain, dynamic_var):
        """Access a nested attribute chain with one segment substituted at runtime.

        Replaces the literal placeholder ``"{var}"`` in ``attr_chain`` with
        ``dynamic_var``, then walks the resulting dotted attribute chain
        step by step (e.g. ``"run_config.{var}.csv"`` with
        ``dynamic_var="temp"`` looks up ``self.run_config.temp.csv``).

        Parameters
        ----------
        attr_chain : str
            Dot-separated attribute path containing the literal substring
            ``"{var}"`` where ``dynamic_var`` should be substituted.
        dynamic_var : str
            Value to substitute for ``"{var}"`` in ``attr_chain``.

        Returns
        -------
        Any
            The value found at the end of the resolved attribute chain.

        Raises
        ------
        AttributeError
            If any segment of the resolved chain doesn't exist.
        """
        # Replace the placeholder {var} with the actual dynamic variable
        attr_chain = attr_chain.replace("{var}", dynamic_var)

        # Split the chain into parts to access attributes step by step
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise AttributeError(f"Attribute '{attr}' not found in the chain '{attr_chain}'.")

        return obj

def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML at {path} must be a mapping.")
    return data


def _find_var_file(var_id: str, base_dir: Path) -> Path:
    for ext in (".yaml", ".yml"):
        p = base_dir / f"{var_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Variable file not found for '{var_id}' in {base_dir}.")


def load_config(yaml_file, top_level_yaml=None, var_dir_name: str = "data_var_configs"):
    """Load a project's main YAML config, merging in palette and per-variable YAML files.

    Resolves a palette dict, then (if the loaded config has a ``data_vars``
    key) resolves and merges each listed variable's own YAML file.

    Palette resolution tries each directory in turn until one has a
    ``palette.yaml``:

    - If ``top_level_yaml`` is given: ``top_level_yaml.parent / var_dir_name``,
      then ``top_level_yaml.parent / <that file's 'data_vars_configs' key>``.
    - Otherwise: ``yaml_file``'s grandparent directory ``/ var_dir_name``,
      then (only if that doesn't contain ``palette.yaml``) its
      great-grandparent directory ``/ var_dir_name``.

    Once found, only the ``pal`` key of that ``palette.yaml`` is merged in
    (existing ``palette_dict`` entries are kept if the key is missing).

    If ``data_vars`` is present, each listed variable id is resolved to a
    ``<var_id>.yaml``/``.yml`` file under ``yaml_file.parent / var_dir_name``
    (see `_find_var_file`). If that file's top level is a single-key
    mapping, it is unwrapped to that key's value (i.e. a file shaped like
    ``{var_id: {...}}`` is treated as just ``{...}``). Each resolved
    variable's dict is merged into ``cfg`` under its own var_id (raising
    ``ValueError`` if that key is already used), and if it has a ``color``
    key, that color is also added to the palette under the var_id.

    Parameters
    ----------
    yaml_file : str or pathlib.Path
        Path to the main project YAML configuration file.
    top_level_yaml : str or pathlib.Path, optional
        Path to an alternate top-level YAML file used only to locate the
        palette directory (see above). Default is ``None``.
    var_dir_name : str, optional
        Directory name where data-variable YAML files (and ``palette.yaml``)
        are stored. Default is ``"data_var_configs"``.

    Returns
    -------
    ProjectConfig
        The merged configuration, with ``file_path`` set to ``yaml_file``'s
        resolved path and a ``pal`` attribute holding the merged palette. If
        any variables were merged in, ``_data_vars_loaded`` holds their ids
        (excluded from `ProjectConfig.to_dict`).
    """
    yaml_path = Path(yaml_file).resolve()
    cfg = _load_yaml(yaml_path)

    palette_dict = cfg.pop("pal", {})
    pal_dir_options = []
    if top_level_yaml is not None:
        top_level_config = _load_yaml(top_level_yaml)

        pal_dir = top_level_yaml.parent /var_dir_name
        pal_dir_options.append(pal_dir)

        pal_dir = top_level_yaml.parent/top_level_config.pop("data_var_configs",'')
        pal_dir_options.append(pal_dir)

    else:
        pal_dir = (yaml_path.parent.parent / var_dir_name).resolve()
        pal_dir_options.append(pal_dir)

        if (pal_dir / 'palette.yaml').exists() is False:
            pal_dir =  yaml_path.parent.parent.parent / var_dir_name
            pal_dir_options.append(pal_dir)

    for pal_dir in pal_dir_options:
        pal_path = (pal_dir / 'palette.yaml')
        if pal_path.exists():
            # Load defensively; if 'pal' key missing, preserve existing palette_dict
            pal_data = _load_yaml(pal_path)
            if isinstance(pal_data, dict) and 'pal' in pal_data:
                palette_dict = pal_data.get('pal', palette_dict)
            # Stop searching once we've found a usable palette file
            break


    dv = cfg.pop("data_vars", None)
    if dv:
        var_ids = list(dv.values()) if isinstance(dv, dict) else list(dv)
        var_dir = (yaml_path.parent / var_dir_name).resolve()

        for var_id in var_ids:
            var_path = _find_var_file(var_id, var_dir)
            var_dict = _load_yaml(var_path)

            # NEW: unwrap if the file is {var_id: {...}} or {alias: {...}} with 1 key
            if len(var_dict) == 1:
                [(only_key, only_val)] = var_dict.items()
                if isinstance(only_val, dict) and (only_key == var_id or True):
                    # Prefer exact match; otherwise still unwrap the single mapping
                    var_dict = only_val

            if var_id in cfg:
                raise ValueError(f"Top-level key '{var_id}' already exists in main config.")
            cfg[var_id] = var_dict
            if 'color' in var_dict:
                palette_dict[var_id] = var_dict['color']

        cfg["_data_vars_loaded"] = var_ids

    cfg['pal'] = palette_dict

    return ProjectConfig(cfg, file_path=str(yaml_path))


def add_var(config, var_type, var_id, var_meta):
    """Add or update a variable entry in a config dict, in place.

    Mutates ``config`` in four ways: (1) sets/updates ``config[var_id]``
    with ``var_meta`` merged onto any existing block (core fields
    ``data_var``/``unit``/``var``/``var_label``/``var_name`` are
    stringified and overwritten; other ``var_meta`` keys are only added if
    not already present); (2) appends ``var_id`` to
    ``config[f"{var_type}_var_ids"]`` (creating it if absent); (3) updates
    ``config[var_type]`` (a group dict with ``ids``/``var``/``alias``/
    ``long_label``), appending ``var_id`` to its ``ids`` list; (4) sets
    ``config["vars"][var_type]`` to the variable's ``var`` name.

    Parameters
    ----------
    config : dict
        Config dict loaded from YAML. Mutated in place.
    var_type : {'col', 'target'}
        Category of variable being added.
    var_id : str
        Key for this variable's block in ``config``.
    var_meta : dict
        Fields to set/overwrite in the variable's block. Must include
        ``'var'`` (required by the group-entry default on first use).
    """
    assert var_type in {"col", "target"}, f"Unknown var_type: {var_type}"

    # Start from existing block if present
    var_block = config.get(var_id, {}).copy()
    # Overwrite core fields
    for field in ("data_var", "unit", "var", "var_label", "var_name"):  # extend as needed
        if field in var_meta:
            var_block[field] = f"{var_meta[field]}"
    # Include any additional metadata
    for key, val in var_meta.items():
        if key not in var_block:
            var_block[key] = val
    # Save updated block
    config[var_id] = var_block

    # Register var_id in var_ids list
    ids_key = f"{var_type}_var_ids"
    config.setdefault(ids_key, [])
    if var_id not in config[ids_key]:
        config[ids_key].append(var_id)

    # Update group entry
    group = config.get(var_type, {"ids": [], "var": var_block["var"],
                                         "alias": var_block.get("alias", var_block["var"]),
                                         "long_label": var_block.get("long_label", var_block.get("var_label"))})
    # setdefault(var_type, )
    # Refresh group metadata
    # group.update({
    #     "var": var_block["var"],
    #     "alias": var_block["var"] if group["alias"] is None else var_block["var"],
    #     "long_label": var_block.get("var_label")
    # })
    if var_id not in group["ids"]:
        group["ids"].append(var_id)

    # Flat vars mapping
    config.setdefault("vars", {})[var_type] = var_block["var"]


def load_proj_config(cfg: Any) -> ProjectConfig:
    """Coerce ``cfg`` into a ``ProjectConfig``, dispatching on its type.

    Parameters
    ----------
    cfg : ProjectConfig or str or pathlib.Path or Mapping
        Already a ``ProjectConfig`` (returned as-is); or a path to a
        ``.yaml``/``.yml`` file (loaded via `load_config`) or a
        ``.json`` file (loaded and wrapped directly); or a mapping (wrapped
        directly, without the YAML-specific palette/data-vars merging that
        `load_config` does).

    Returns
    -------
    ProjectConfig

    Raises
    ------
    ValueError
        If ``cfg`` is a path with an unsupported file extension.
    TypeError
        If ``cfg`` is none of the supported types.
    """
    if isinstance(cfg, ProjectConfig):
        return cfg
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return load_config(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            return ProjectConfig(data)
        raise ValueError(f"Unsupported config file type: {path}")
    if isinstance(cfg, Mapping):
        return ProjectConfig(dict(cfg))
    raise TypeError("cfg must be a ProjectConfig, dict, or path to YAML/JSON")
