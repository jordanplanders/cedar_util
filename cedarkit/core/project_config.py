from __future__ import annotations

import json
from typing import Any, Mapping

import yaml
from pathlib import Path

try:
    import cedarkit.utils.routing.paths

except ImportError:
    # Fallback: imports when running as a package
    import utils.paths


class ProjectConfig:
    def __init__(self, config_data, file_path=None, is_root=True):
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
        return f"{self.__class__.__name__}({self.__dict__})"

        # Function to check for nested attributes
    def has_nested_attribute(self, attr_chain):
        attrs = attr_chain.split('.')
        obj = self
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return False
        return True
    # Function to add or update an attribute
    def add_attribute(self, key, value):
        if isinstance(value, dict):
            setattr(self, key, ProjectConfig(value))
        else:
            setattr(self, key, value)

    # Function to add an item to a list
    def add_to_list(self, list_name, item):
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
        return getattr(self, "_entry", None)

    def set_consolidated_format(self, fmt: str):
        """Override consolidated output format (default: parquet for csv entry)."""
        self._consolidated_format = str(fmt) if fmt is not None else None

    def get_consolidated_format(self):
        return getattr(self, "_consolidated_format", None)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if key in {"file_path", "_data_vars_loaded"}:
                continue
            if isinstance(value, ProjectConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    # Function to save the configuration back to the YAML file
    def save_config(self):
        if self.file_path is None:
            raise ValueError("No file path specified for saving the configuration.")
        with open(self.file_path, 'w') as file:
            yaml.dump(self.to_dict(), file)

    # Function to dynamically access nested attributes
    def get_dynamic_attr(self, attr_chain, dynamic_var):
        """
        Accesses a nested attribute dynamically where part of the chain is variable.

        Parameters:
        - attr_chain: The attribute chain with a placeholder for the dynamic part (e.g., "run_config.{var}.csv")
        - dynamic_var: The variable part that replaces the placeholder

        Returns:
        - The requested attribute value or raises an error if not found
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
    ''''
    Load project configuration from a YAML file, including data variable definitions.
    Parameters:
        - yaml_file: Path to the main project YAML configuration file.
        - var_dir_name: Directory name where data variable YAML files are stored.
    Returns:
        - ProjectConfig instance with loaded configuration.

    Notes:
        - Data variable YAML files are expected to be in the specified var_dir_name directory,
            located relative to the main YAML file.
        - If a palette.yaml file exists in a parent var_dir_name directory, its contents
            will be merged into the main palette dictionary.
    '''
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
    """
    Add or update a variable entry in the config using a template block.

    Parameters:
    - config: dict loaded from YAML
    - var_type: 'col' or 'target'
    - var_id: key of the variable in config
    - var_meta: dict with fields to overwrite in the variable block
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
