import copy
import os
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from cedarkit.utils.cli import log_line

def check_location(target_path=None, hpc_word='lplander'):
    """Classify a path as local or HPC based on a marker substring.

    Parameters
    ----------
    target_path : str or pathlib.Path, optional
        Path to inspect. Default is the current working directory
        (``pathlib.Path.cwd()``).
    hpc_word : str, default 'lplander'
        Substring whose presence in ``target_path`` marks it as an HPC path
        (e.g. a username or cluster mount point that only appears in HPC
        filesystem layouts).

    Returns
    -------
    str
        ``'hpc'`` if ``hpc_word`` appears in ``target_path``, otherwise
        ``'local'``.

    Notes
    -----
    The result selects which location-specific block of the project config
    (e.g. ``config.local`` vs. ``config.hpc``) other routing helpers such as
    ``set_calc_path`` read from.

    See Also
    --------
    set_calc_path : Uses this to select the calc-directory config block.
    """
    if target_path is None:
        target_path = Path.cwd()

    if hpc_word in str(target_path):
        return 'hpc'
    else:
        return 'local'


def set_calc_path(args, proj_dir, config, second_suffix=''):
    """Resolve the calculation-output directory for a run.

    Uses an explicit CLI override if given, otherwise reads the
    location-specific (``local``/``hpc``) ``calc_dir`` setting from
    ``config``, selected via ``check_location``.

    Parameters
    ----------
    args : argparse.Namespace or None
        Parsed CLI arguments (see ``get_parser``). If ``args.calc_dir`` is
        set, it is used directly (as ``proj_dir / args.calc_dir``) and
        ``config``/``second_suffix`` are ignored.
    proj_dir : pathlib.Path
        Project root directory.
    config : cedarkit.core.project_config.ProjectConfig
        Project configuration, consulted for the location-specific
        ``calc_dir`` value when ``args`` doesn't override it.
    second_suffix : str, default ''
        Appended to the config-derived directory name — useful for keeping
        parallel run variants (e.g. a test run) in separate directories.

    Returns
    -------
    pathlib.Path
        The resolved calculation directory (not guaranteed to exist yet).

    See Also
    --------
    check_location : Determines which config block ``calc_dir`` is read from.
    set_output_path : Resolves the output directory nested under this path.
    """
    calc_location = None
    if args is not None:
        if args.calc_dir is not None:
            calc_location = proj_dir/args.calc_dir

    if calc_location is None:
        loc = check_location(proj_dir)
        calc_dir = config.get_dynamic_attr("{var}.calc_dir", loc)
        calc_location = proj_dir / (calc_dir + f'{second_suffix}')  #'calc_local_tmp'


    return calc_location


def _entry_format(cfg):
    entry = getattr(cfg, "get_entry", lambda: None)()
    return entry or None


def _consolidated_format(cfg, entry):
    override = getattr(cfg, "get_consolidated_format", lambda: None)()
    if override:
        return override
    if entry == "csv":
        return "parquet"
    if entry == "sqlite":
        return "sqlite"
    return None


def _fmt_block(cfg, block_name, fmt):
    if fmt is None:
        return None
    if not hasattr(cfg, block_name):
        return None
    block = getattr(cfg, block_name)
    if not hasattr(block, fmt):
        return None
    return getattr(block, fmt)


def _join_dir_structure(base: Path, dir_structure: str) -> Path:
    try:
        # Avoid duplicating if dir_structure already starts with base leaf
        if dir_structure and Path(dir_structure).parts and Path(dir_structure).parts[0] == base.name:
            return base / Path(*Path(dir_structure).parts[1:])
    except Exception:
        pass
    return base / dir_structure if dir_structure else base


def resolve_intermediate_dir(calc_location: Path, cfg, fmt: str, include_dir_structure: bool = False) -> Path:
    base = Path(calc_location) / "intermediate"
    block = _fmt_block(cfg, "intermediate", fmt)
    if block is None:
        # legacy: fall back to output format block
        block = _fmt_block(cfg, "output", fmt)
    if block is None:
        return base / fmt
    fmt_dir = getattr(block, "dir", fmt) or fmt
    fmt_base = base / fmt_dir
    if not include_dir_structure:
        return fmt_base
    dir_structure = getattr(block, "dir_structure", "")
    return _join_dir_structure(fmt_base, str(dir_structure) if dir_structure is not None else "")


def resolve_consolidated_dir(
    calc_location: Path,
    cfg,
    fmt: str | None,
    *,
    include_dir_structure: bool = False,
) -> Path:
    if fmt is None or fmt == "auto":
        entry = _entry_format(cfg)
        fmt = _consolidated_format(cfg, entry) or entry or "sqlite"
    elif fmt == "sqlite":
        # Honor consolidated format overrides even when sqlite is requested.
        entry = _entry_format(cfg)
        fmt = _consolidated_format(cfg, entry) or fmt

    loc = check_location(calc_location)
    output_sub = cfg.get_dynamic_attr("{var}.output_dir", loc)

    base = Path(calc_location) / str(output_sub)
    block = _fmt_block(cfg, "output", fmt)
    if block is None:
        return base / fmt

    fmt_dir = getattr(block, "dir", fmt) or fmt
    fmt_base = base / fmt_dir
    if not include_dir_structure:
        return fmt_base

    dir_structure = getattr(block, "dir_structure", "")
    return _join_dir_structure(fmt_base, str(dir_structure) if dir_structure is not None else "")


def set_output_path(args, calc_location, config):
    """Resolve the output directory nested under a calculation location.

    Parameters
    ----------
    args : argparse.Namespace or None
        Parsed CLI arguments. If ``args.output_dir`` is set, it is used
        directly (as ``calc_location / args.output_dir``) and ``config`` is
        ignored.
    calc_location : pathlib.Path
        Base calculation directory, typically from ``set_calc_path``.
    config : cedarkit.core.project_config.ProjectConfig
        Project configuration; ``config.hpc.output_dir`` is used when
        ``args`` doesn't override it.

    Returns
    -------
    pathlib.Path or None
        The resolved output directory, or ``None`` if neither ``args`` nor
        ``config.hpc.output_dir`` provides one (a message is printed to
        stdout in that case rather than raising).

    See Also
    --------
    set_calc_path : Resolves the ``calc_location`` this builds on.
    """
    output_dir = None
    if args is not None:
        if args.output_dir is not None:
            output_dir = calc_location / args.output_dir

    if output_dir is None:
        try:
            output_dir = calc_location / config.hpc.output_dir

        except AttributeError:
            print('AttributeError: config.runners.output_dir not found', file=sys.stdout, flush=True)


    return output_dir


def sqlite_paths(proj_dir, config, *, run_id=None, calc_location=None, output_dir=None, ensure=True):
    import warnings
    warnings.warn(
        "sqlite_paths is deprecated; use resolve_intermediate_dir/resolve_consolidated_dir instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if output_dir is None:
        if calc_location is None:
            calc_location = set_calc_path(None, proj_dir, config, second_suffix="")
        output_dir = set_output_path(None, calc_location, config)

    if not hasattr(config, "output") or not hasattr(config.output, "sqlite"):
        raise AttributeError("config.output.sqlite is required to build sqlite paths.")

    entry = _entry_format(config)
    consolidated_fmt = _consolidated_format(config, entry) or "sqlite"
    if hasattr(config, "intermediate") and hasattr(config.intermediate, "sqlite"):
        sqlite_dir = resolve_intermediate_dir(Path(calc_location), config, "sqlite", include_dir_structure=True)
    else:
        # legacy behavior: run dbs under output dir
        sqlite_cfg = config.output.sqlite
        dir_structure = getattr(sqlite_cfg, "dir_structure", "run_dbs")
        sqlite_dir = Path(output_dir) / dir_structure
    if ensure:
        sqlite_dir.mkdir(parents=True, exist_ok=True)

    sqlite_cfg = config.output.sqlite
    collector_name = getattr(sqlite_cfg, "consolidated_db", None)
    if not collector_name:
        collector_name = getattr(sqlite_cfg, "file_format", "collector")
    collector_name = collector_name or "collector"
    def _nested_attr(obj, path, default=None):
        cur = obj
        for part in path.split("."):
            if cur is None or not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
        return cur

    # Prefer dyad config IDs when available, then human-readable labels, then
    # legacy graphccm-style config.data.var_A/var_B fallbacks.
    col_var_id = (
        _nested_attr(config, "col.var_id")
        or _nested_attr(config, "col.var")
        or _nested_attr(config, "vars.col")
        or _nested_attr(config, "data.var_A")
        or ""
    )
    target_var_id = (
        _nested_attr(config, "target.var_id")
        or _nested_attr(config, "target.var")
        or _nested_attr(config, "vars.target")
        or _nested_attr(config, "data.var_B")
        or ""
    )

    collector_name = template_replace(
        str(collector_name),
        {
            "col_var_id": str(col_var_id),
            "target_var_id": str(target_var_id),
            "run_id": str(run_id or ""),
        },
        return_replaced=False,
    )
    consolidated_dir = resolve_consolidated_dir(Path(calc_location), config, consolidated_fmt)
    if ensure:
        consolidated_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(consolidated_dir.glob("collector*.sqlite"))
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple collector SQLite files detected in resolved consolidated directory. "
            f"Candidates: {[str(p) for p in candidates]}"
        )
    collector_path = consolidated_dir / f"{collector_name}.sqlite"

    run_db_path = None
    if run_id is not None:
        run_cfg = getattr(getattr(config, "intermediate", None), "sqlite", sqlite_cfg)
        file_format = getattr(run_cfg, "file_format", "run_{run_id}")
        try:
            run_file = template_replace(file_format, {"run_id": run_id}, return_replaced=False)
        except Exception:
            run_file = file_format.format(run_id=run_id)
        run_db_path = sqlite_dir / f"{run_file}.sqlite"

    return sqlite_dir, collector_path, run_db_path


def set_grp_path(output_path, d, config=None, source='csv', grp_level='grp_dir_structure', make_grp=True):
    if 'tp' in d.keys():
        d['Tp'] = d['tp']
    tmp_d = d.copy()
    tmp_d= {k:v[0] if isinstance(v, list) and len(v)==1 else v for k,v in tmp_d.items()}
    # config_path = parent_path#set_model_config_path(parent_path, d, config=config)

    if config is not None:
        output_config = config.get_dynamic_attr("output.{var}", source)
        grp_path_template = output_config.dir_structure
        grp_path_template_filled = template_replace(grp_path_template, tmp_d, return_replaced=False)
        grp_path = output_path / grp_path_template_filled
        # print('grp_path_template_filled', grp_path, file=sys.stdout, flush=True)

    # if source == 'csv':
    #     if 'lag' in d and d['lag'] is not None:
    #         grp_level = 'dir_structure_csv'
    #     else:
    #         grp_level = 'grp_dir_structure'
    #     # grp_level = 'dir_structure_csv'
    # else:
    #     grp_level = 'dir_structure'
    #
    # if config is not None:
    #     try:
    #         grp_path_template = config.get_dynamic_attr("output.{var}", grp_level) # config.output.grp_dir_structure
    #         grp_path_template_filled = template_replace(grp_path_template,tmp_d, return_replaced=False)
    #         grp_path = output_path / grp_path_template_filled
    #
    #     except:
    #         pass
    else:
        grp_path_template_filled = f'{d["col_var_id"]}_{d["target_var_id"]} / E{d["E"]}_tau{d["tau"]}'
        grp_path = output_path / grp_path_template_filled


    if make_grp is True:
        grp_path.mkdir(exist_ok=True, parents=True)

    # print('grp_path', grp_path, file=sys.stdout, flush=True)
    return grp_path


def set_proj_dir(proj_name, current_path):
    if proj_name in str(current_path):
        proj_dir = Path(str(current_path).split(proj_name)[0]) / proj_name
    else:
        proj_dir = current_path / proj_name

    if 'proj_config.yaml' in os.listdir(proj_dir):
        return proj_dir
    else:
        print('proj_config.yaml not found in project directory', file=sys.stdout, flush=True)
        print('proj_config.yaml not found in project directory', file=sys.stderr, flush=True)
        sys.exit(0)


def set_model_config_path(parent_path, d, config=None):

    if config is not None:
        try:
            dir_structure = config.output.non_grp_structure
            dir_structure_filled = dict_replace(dir_structure, d)
            return parent_path / dir_structure_filled
        except:
            pass
    return parent_path / f'knn_{d["knn"]}' / f'tp_{d["Tp"]}' / f'lag_{d["lag"]}'


def dict_replace(template, d):
    if isinstance(d, dict) is False:
        print('Error: d must be a dictionary', template, d, file=sys.stdout, flush=True)
        return template
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
    return template


def template_replace(template, d, return_replaced=True):
    replaced = []
    old_template = copy.copy(template)
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
        if template != old_template:
            replaced.append(key)
            old_template = copy.copy(template)
    if return_replaced is False:
        return template

    return template, replaced


def check_exists(file_name, calc_dir):
    dir_exists = os.path.exists(str(calc_dir))
    if dir_exists is False:
        return False, False

    calc_dir_list = os.listdir(str(calc_dir))#[entry.name for entry in calc_dir.iterdir() if entry.is_dir()]
    pset_id = file_name.split('_E')[0]  # assuming pset_id is the first part of the file name
    stem = file_name.split(pset_id)[1].lstrip('_')  # assuming the pset_id is before the first '__'
    pset_files = [fn for fn in calc_dir_list if fn.startswith(pset_id)]
    stem_files = [fn for fn in pset_files if fn.endswith(stem)]

    pset_exists = False
    stem_exists = False
    if str(file_name) in calc_dir_list:
        pset_exists = True
    if len(stem_files) > 0:
        stem_exists = True

    return pset_exists, stem_exists


def resolve_dyad_dir(*, project_dir: Path, left_id: str, right_id: str) -> Path:
    candidates = [
        Path(project_dir) / 'dyads' / f'{left_id}_{right_id}',
        Path(project_dir) / 'dyads' / f'{right_id}_{left_id}',
        Path(project_dir) / f'{left_id}_{right_id}',
        Path(project_dir) / f'{right_id}_{left_id}',
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not resolve dyad directory for pair: {left_id}, {right_id}')
