
import collections.abc
from copy import deepcopy
import pandas as pd
from functools import reduce
import operator
from collections import defaultdict
import uuid
import gc
import numpy as np
from types import SimpleNamespace
import polars as pl

# import cedarkit.utils.paths
# from cedarkit.utils.paths import set_calc_path, set_output_path, template_replace, check_exists

from cedarkit.core.data_var import *
from cedarkit.core.relationship import *
from cedarkit.utils.workflow.process_output import infer_relation_variables
from cedarkit.utils.routing import *
from cedarkit.utils.routing import template_replace
from cedarkit.utils.cli import log_line

# dump
import os
from pathlib import Path
# SCRIPT = Path(__file__).resolve().name
import logging
logger = logging.getLogger(__name__)


def resolve_dyad_anchored_path(path, dyad_dir, *, prefer_local=False):
    """Return an existing local equivalent of ``path`` under ``dyad_dir``.

    Output grids have historically stored absolute paths.  Those paths are
    valid only on the machine where the grid was created, but the part below
    the dyad directory is portable.  For example, this maps
    ``/hpc/.../dyads/A_B/tmp/result.parquet`` to
    ``/local/.../dyads/A_B/tmp/result.parquet`` when ``dyad_dir`` is the
    local ``A_B`` directory.  When ``prefer_local`` is ``True``, that local
    counterpart wins even when the original path remains accessible.  The
    original path is returned when no existing local counterpart can be
    confirmed.
    """
    if path is None:
        return None

    original = Path(path)
    if dyad_dir is None:
        return original

    dyad_dir = Path(dyad_dir)
    if not dyad_dir.is_dir():
        return original

    # Use the final occurrence: a parent directory can occasionally share
    # the dyad's name, while the final one is the actual anchor.
    anchor_positions = [
        index for index, part in enumerate(original.parts)
        if part == dyad_dir.name
    ]
    for anchor_index in reversed(anchor_positions):
        candidate = dyad_dir.joinpath(*original.parts[anchor_index + 1:])
        if candidate.exists() and (prefer_local or not original.exists()):
            return candidate
    return original


def _dyad_dir_from_tmp(tmp_dir):
    """Infer a dyad directory from CedarKit's conventional ``<dyad>/tmp``."""
    if tmp_dir is None:
        return None
    tmp_path = Path(tmp_dir)
    return tmp_path.parent if tmp_path.name == "tmp" else None


def correct_iterable(obj):
    """Normalize a value into a list (or ``None``).

    Strings are treated as a single scalar, not as an iterable of
    characters. Non-iterable, non-string values are wrapped in a
    single-element list.

    Parameters
    ----------
    obj : Any
        Value to normalize.

    Returns
    -------
    list or None
        ``None`` if ``obj`` is ``None``; ``[obj]`` if ``obj`` is a string or
        not iterable; ``list(obj)`` otherwise.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return [obj]
    else:
        if isinstance(obj, collections.abc.Iterable):
            return list(obj)
        else:
            return [obj]

def get_static(obj):
    """Extract a single static value from a possibly-iterable trait value.

    The approximate inverse of :func:`correct_iterable`: collapses a
    single-element iterable back to its scalar, but treats genuinely
    multi-valued (non-static) iterables as having no single answer.

    Parameters
    ----------
    obj : Any
        Value to collapse.

    Returns
    -------
    Any or None
        ``None`` if ``obj`` is ``None``; ``obj`` unchanged if it's a string
        or not iterable; the sole element if ``obj`` is a length-1
        iterable; ``None`` if ``obj`` is an iterable with any other length
        (including zero).
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    else:
        if isinstance(obj, collections.abc.Iterable):
            if len(obj) == 1:
                return obj[0]
            else:
                return None
        else:
            return obj


def extract_from_pattern(filename: str, pattern_str: str):
    """Extract integer parameter values from a filename via a ``{name}`` template.

    Each ``{name}`` placeholder in ``pattern_str`` is converted to a named
    regex group matching an optional-sign integer (``-?\\d+``), then matched
    against ``filename``. Relies on the module-level ``re`` name brought in
    by one of this module's ``from ... import *`` statements rather than an
    explicit ``import re``.

    Parameters
    ----------
    filename : str
        Filename to extract values from.
    pattern_str : str
        Template containing ``{name}`` placeholders, e.g.
        ``"E{E}_tau{tau}_lag{lag}"``.

    Returns
    -------
    dict[str, int]
        One entry per placeholder, e.g.
        ``extract_from_pattern("E4_tau1_lag-5.parquet", "E{E}_tau{tau}_lag{lag}")``
        returns ``{'E': 4, 'tau': 1, 'lag': -5}``.

    Raises
    ------
    ValueError
        If ``filename`` doesn't match the pattern derived from ``pattern_str``.
    """
    # Convert format specifiers like {E}, {tau}, {lag} into named regex groups
    regex = re.sub(r"\{(\w+)\}",
                   lambda m: f"(?P<{m.group(1)}>-?\\d+)", pattern_str)

    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match pattern '{pattern_str}'")

    # Convert all extracted values to integers
    return {k: int(v) for k, v in match.groupdict().items()}


def compute_delta_rho_grp(
        lag_tbl,
        gd: dict,
        # *,
        stats: bool = True,
        full: bool = False,
        best_window_halfwidth: int = 15,
        min_window: int = 30,
        max_window: int = 50,
        rng_seed: int = 1,
        annotation: str = ""
):
    """Compute delta-rho statistics and full vectors from a lagged correlation table.

    For one group's CCM output (rows varying by ``LibSize``), computes:

    - mean ``rho`` in the min-libsize band (``LibSize < min(LibSize) + min_window``)
    - mean ``rho`` in the max-libsize band (``LibSize > max(LibSize) - max_window``)
    - the best libsize (the ``LibSize`` value with the highest mean ``rho``)
    - mean ``rho`` in a window around the best libsize
      (``best_libsize +/- best_window_halfwidth``)
    - delta rho = max-libsize-band mean rho minus min-libsize-band mean rho
    - (if ``full``) bootstrap-style paired samples (with replacement) of the
      above per-row values, for downstream distributional analysis

    Used by ``OutputCollection.calc_delta_rho`` (called ``OutputGrp`` in
    earlier versions of this docstring; see :class:`OutputCollection`).

    Parameters
    ----------
    lag_tbl : polars.LazyFrame or polars.DataFrame or pandas.DataFrame
        Lagged correlation data for one group. Must have ``'LibSize'``
        (int/float) and ``'rho'`` (float) columns.
    gd : dict
        Group descriptors (e.g. trait values identifying this group) copied
        into each output row/column via :func:`get_static`.
    stats : bool, optional
        Whether to compute the summary statistics table. Default is ``True``.
    full : bool, optional
        Whether to compute the full bootstrap-sampled vectors table. Default
        is ``False``.
    best_window_halfwidth : int, optional
        Half-width of the window around the best libsize. Default is ``15``.
    min_window : int, optional
        Width of the min-libsize band. Default is ``30``.
    max_window : int, optional
        Width of the max-libsize band. Default is ``50``.
    rng_seed : int, optional
        Seed for the bootstrap sampling RNG. Default is ``1``.
    annotation : str, optional
        Free-text label copied into output rows. Default is ``""``.

    Returns
    -------
    tuple[polars.LazyFrame or None, polars.LazyFrame or None]
        ``(stats_tbl, full_tbl)``. ``stats_tbl`` is ``None`` unless ``stats``
        is ``True``; ``full_tbl`` is ``None`` unless ``full`` is ``True``.
        Both are ``None`` if ``lag_tbl`` is empty.

    Raises
    ------
    TypeError
        If ``lag_tbl`` is not one of the supported table types.
    """
    # lag_tbl = self.table.full
    if isinstance(lag_tbl, pl.LazyFrame):
        lag_df = lag_tbl.collect().to_pandas()
    elif isinstance(lag_tbl, pl.DataFrame):
        lag_df = lag_tbl.to_pandas()
    elif isinstance(lag_tbl, pd.DataFrame):
        lag_df = lag_tbl.copy()
    else:
        raise TypeError(f"Unsupported lag_tbl type: {type(lag_tbl)}")

    if lag_df is None or len(lag_df) == 0:
        log_line(logger, 'empty lag_tbl', indent=0,
                 log_type="info")
        return (None, None)

    lib = lag_df['LibSize']

    # thresholds at ends
    lib_min = lib.min()
    lib_max = lib.max()

    # min/max libsize bands
    min_mask = lib < (lib_min + min_window)
    max_mask = lib > (lib_max - max_window)

    min_tbl = lag_df[min_mask].copy()
    max_tbl = lag_df[max_mask].copy()

    gb = lag_df.groupby("LibSize", as_index=False)["rho"].mean()
    gb_sorted = gb.sort_values("rho", ascending=False)
    best_libsize = gb_sorted["LibSize"].iloc[0]

    # window around best libsize
    lo = best_libsize - best_window_halfwidth
    hi = best_libsize + best_window_halfwidth
    win_mask = (lib >= lo) & (lib <= hi)
    best_tbl = lag_df[win_mask].copy()
    # stats
    stats_tbl = None

    n_min = len(min_tbl)
    n_max = len(max_tbl)
    n_best = len(best_tbl)
    sample_size = max(n_min, n_max)

    rng = np.random.default_rng(rng_seed)
    # sample indices with replacement from each subset
    idx_min = rng.integers(0, n_min, size=sample_size) if n_min > 0 else np.array([], dtype=np.int64)
    idx_max = rng.integers(0, n_max, size=sample_size) if n_max > 0 else np.array([], dtype=np.int64)
    idx_best = rng.integers(0, n_best, size=sample_size) if n_best > 0 else np.array([], dtype=np.int64)

    if n_min > 0:
        min_rhos = min_tbl['rho'].to_numpy()[idx_min]
        min_rhos = np.maximum(min_rhos, 0)
    else:
        min_rhos = np.full(sample_size, np.nan, dtype=float)

    if n_max > 0:
        max_rhos = max_tbl['rho'].to_numpy()[idx_max]
    else:
        max_rhos = np.full(sample_size, np.nan, dtype=float)

    delta_rho_vec = max_rhos - min_rhos

    # also expose the raw rho values from the "best window"
    best_rhos = best_tbl['rho'].to_numpy()[idx_best] if n_best > 0 else np.full(sample_size, np.nan, dtype=float)

    if stats:
        best_mean_rho = best_tbl['rho'].mean() if len(best_tbl) > 0 else np.nan
        min_mean_rho = min_tbl['rho'].mean() if len(min_tbl) > 0 else np.nan
        max_mean_rho = max_tbl['rho'].mean() if len(max_tbl) > 0 else np.nan
        delta_rho = (max_mean_rho - min_mean_rho) if (
                    np.isfinite(max_mean_rho) and np.isfinite(min_mean_rho)) else np.nan

        stats_row = {}
        for k, v in gd.items():
            stats_row[k] = [get_static(v)]

        stats_row['maxrho'] = [best_mean_rho]
        stats_row['minlibsize_rho'] = [min_mean_rho]
        stats_row['maxlibsize_rho'] = [max_mean_rho]
        stats_row['delta_rho'] = [delta_rho]
        stats_row['annotation'] = [annotation]

        stats_tbl = pl.from_pandas(pd.DataFrame(stats_row)).lazy()

    # full vectors with bootstrap-style paired sampling (with replacement)
    full_tbl = None
    if full:
        cols_full = {
            'minlibsize_rho': min_rhos.tolist(),
            'maxlibsize_rho': max_rhos.tolist(),
            'delta_rho': delta_rho_vec.tolist(),
            'maxrho': best_rhos.tolist(),
            'annotation': [annotation] * sample_size,
        }
        for k, v in gd.items():
            cols_full[k] = [get_static(v)] * sample_size

        full_tbl = pl.from_pandas(pd.DataFrame(cols_full)).lazy()

    return stats_tbl, full_tbl


#########################################
class RunConfig:
    """Configuration (CCM parameters + variable identities) for one run or group of runs.

    Holds the CCM embedding/prediction parameters (``E``, ``tau``, ``lag``,
    ``knn``, ``Tp``/``tp``, training-index bounds), which two variables are
    being related (``col_var``/``target_var`` and their resolved
    ``*_var_obj`` objects), surrogate-run bookkeeping (``surr_num``,
    ``surr_var``), and output file location. The set of "trait" attributes
    (see :attr:`traits`) is what gets used for filtering/grouping output
    files elsewhere in this module.

    A single ``RunConfig`` can represent either one fully-specified run, or
    a group of runs sharing some traits (when a "trait" attribute is set to
    a list rather than a scalar — see :func:`correct_iterable`/
    :func:`get_static`). Subclassed by :class:`CMConfigBase`
    (`CCMConfig`'s base) and used directly as `DataGroup.parent_config`.
    """

    def __init__(self, grp_d, tmp_dir=None, **_ignored_kwargs):
        """Initialize all known trait attributes to ``None``/defaults, then populate from ``grp_d``.

        Parameters
        ----------
        grp_d : dict
            Trait values to set, passed to :meth:`populate`.
        tmp_dir : str or pathlib.Path, optional
            Temporary directory for intermediate files. Default is ``None``.
        **_ignored_kwargs
            Accepted and discarded, so callers can pass extra keys without
            raising.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.E = None
        self.tau = None
        self.lag = None
        self.train_ind_i = 0
        self.train_ind_f = -1
        self.knn = None

        # eventually factor out Tp in favor of tp
        self.Tp = None
        self.tp = None
        
        self.sample = None
        self.weighted = None

        self.col_var_id=None
        self.col_var=None
        self.col_var_obj=None

        self.target_var_id=None
        self.target_var=None
        self.target_var_obj=None

        self.am_id=None
        self.am_num=None
        self.surr_num=None
        self.surr_var=None
        self.output_path=None
        self.output_format='parquet'
        self.output_query=None
        self.output_params=None
        self.relation=None

        self.pset_id=None
        # self.train_ind_i = None
        # self.train_ind_f = None

        self.populate(grp_d)
        self.tmp_dir = tmp_dir

        self.proj_dir = None

        if self.train_ind_i is None:
            self.train_ind_i = 0
            self.train_ind_i = int(self.train_ind_i)
        if self.train_ind_f is None:
            self.train_ind_f = -1
            self.train_ind_f = int(self.train_ind_f)

        # self.exclusion_radius = np.abs(self.tau * (self.E - 1))

    def populate(self, grp_d):
        """Set trait attributes on this instance from ``grp_d``, in place.

        Only keys that already exist as attributes on ``self`` are set
        (unknown keys in ``grp_d`` are silently ignored). Before that,
        ``Tp``/``tp`` are reconciled: whichever of the two is non-``None``
        in ``grp_d`` (preferring ``Tp``) is written back into ``grp_d``
        under *both* keys, so they end up equal regardless of which name
        the caller used (see the ``# eventually factor out Tp in favor of
        tp`` note in :meth:`__init__`). If ``self.pset_id`` is still unset
        after that, it falls back to ``grp_d.get('id')``.

        Parameters
        ----------
        grp_d : dict
            Trait values to set. Mutated in place (the ``Tp``/``tp``
            reconciliation above).
        """
        # print('Populating RunConfig with grp_d:', grp_d, file=sys.stdout, flush=True)
        # print_log_line(SCRIPT, inspect.currentframe().f_code.co_name, ['Populating RunConfig with grp_d:', grp_d], level=1, log_type='info')
        log_line(
            self.log,
            ["Populating RunConfig with grp_d:", grp_d],
            indent=1,
            log_type="debug",  # or "info", but debug is nice for “comment/uncomment” style
        )

        tp = grp_d.get('Tp', None)
        if tp is None:
            tp = grp_d.get('tp', None)
        grp_d['Tp'] = tp
        grp_d['tp'] = tp

        for key, value in grp_d.items():
            if hasattr(self, key):
                setattr(self, key, value)
                log_line( self.log,["Sets RunConfig trait", key, "to", value], indent=2, log_type="debug")
                # print('Sets RunConfig trait', key, 'to', value, file=sys.stdout, flush=True)

        if self.pset_id is None:
            self.pset_id = grp_d.get('id', None)


        log_line(
            self.log,
            ["RunConfig populated traits:", self.to_dict()],
            indent=1,
            log_type="debug",
        )
        # print('RunConfig populated traits:', self.to_dict(), file=sys.stdout, flush=True)



    def copy(self):
        """Return a deep copy of this ``RunConfig`` (including nested objects like ``*_var_obj``)."""
        return deepcopy(self)

    def get_trait_value(self, trait):
        """Return ``getattr(self, trait, None)`` — a non-raising attribute lookup."""
        return getattr(self, trait, None)

    @property
    def var_x(self):
        # Alias for self.col_var (the predictor/"x" variable name).
        return self.col_var

    @property
    def var_y(self):
        # Alias for self.target_var (the target/"y" variable name).
        return self.target_var

    @property
    def var_x_obj(self):
        # Alias for self.col_var_obj.
        return self.col_var_obj

    @property
    def var_y_obj(self):
        # Alias for self.target_var_obj.
        return self.target_var_obj

    @property
    def traits(self):
        """Names of all instance attributes except ``output_path``, ``output_format``, ``output_query``, ``output_params``, and ``log``.

        This is the attribute set used for filtering/grouping output files
        elsewhere (:meth:`to_dict`, :meth:`pull_output`, :meth:`trait_hierarchy`,
        and `DataGroup`/`OutputCollection` methods) — i.e. everything that
        describes *what run this is*, excluding output-location bookkeeping.

        Returns
        -------
        list[str]
        """
        return [key for key in self.__dict__.keys()
                if key not in ['output_path', 'output_format', 'output_query', 'output_params', 'log']]

    def to_dict(self):
        """Return this instance's non-``None`` trait attributes as a dict.

        Returns
        -------
        dict
            ``{key: value for key in self.traits if value is not None}``.
        """
        return {key: value for key, value in self.__dict__.items() if key in self.traits and value is not None}

    def resolve_output_path(self, path, dyad_dir=None, *, prefer_local=False):
        """Resolve a stale absolute output path against this run's dyad.

        ``tmp_dir`` conventionally lives at ``<dyad>/tmp``; that gives old
        pickled ``RunConfig`` instances enough local context to repair a
        path copied from another machine.
        """
        if dyad_dir is None:
            dyad_dir = _dyad_dir_from_tmp(self.tmp_dir)
        return resolve_dyad_anchored_path(path, dyad_dir, prefer_local=prefer_local)

    def pull_output(self, to_table=False, limit_surr=True):
        """Load this run's output file, filtered to this instance's trait values.

        Reads ``self.output_path[0]`` — as a SQLite query (if
        ``self.output_format == 'sqlite'``, using ``self.output_query``/
        ``self.output_params``) or as a Polars Parquet scan filtered by
        :meth:`to_dict`'s trait values otherwise.

        Parameters
        ----------
        to_table : bool, optional
            If ``True``, return the raw table (a ``polars.LazyFrame`` for
            SQLite or a filtered ``polars.DataFrame`` for Parquet) instead of
            wrapping it. Default is ``False``.
        limit_surr : bool, optional
            Currently unused by this method's body.

        Returns
        -------
        polars.DataFrame or OutputCollection or None
            ``None`` if ``self.output_path`` is unset/empty (a message is
            logged in that case rather than raising). Otherwise the raw
            table if ``to_table`` is ``True``, else an ``OutputCollection``
            wrapping it.
        """
        if self.output_path is None or len(self.output_path) == 0:
            print('no output path specified')
            log_line(self.log, 'no output path specified', indent=0, log_type="error")
            return

        file_path = self.resolve_output_path(self.output_path[0])
        self.output_path[0] = file_path
        log_line(self.log, ['pulling from', file_path], indent=0, log_type="info")

        if self.output_format == 'sqlite':
            out = Output(
                full=None,
                path=file_path,
                query=self.output_query,
                params=self.output_params,
                format='sqlite',
                tmp_dir=self.tmp_dir,
            )
            table = out.table
            if to_table:
                return table
            return OutputCollection(in_table=table, grp_specs=self, outtype='full', tmp_dir=self.tmp_dir)

        all_traits = self.to_dict()

        lf = pl.scan_parquet(str(file_path))
        schema_names = set(lf.collect_schema().names())
        filter_exprs = [
            pl.col(key).is_in(correct_iterable(value))
            for key, value in all_traits.items()
            if value is not None and key in schema_names
        ]
        if filter_exprs:
            lf = lf.filter(reduce(operator.and_, filter_exprs))
        filtered_table = lf.collect()

        if to_table is True:
            return filtered_table
        else:
            return OutputCollection(in_table=filtered_table, grp_specs=self, outtype='full', tmp_dir=self.tmp_dir)

    def trait_hierarchy(self, full_ds, trait, level="below", threshold=0.9, include_ids=False):
        """
        Return traits that are above or below the grouping level of a given trait.

        Parameters
        ----------
        full_ds : pandas.DataFrame or polars.DataFrame or polars.LazyFrame
            The dataset to analyze. Converted to a pandas DataFrame internally.
        trait : str
            The reference column defining the grouping level.
        level : {'below', 'above'}, default 'below'
            Whether to return traits that vary below (within groups)
            or remain constant above (across groups).
        threshold : float, default 0.9
            Fraction of uniqueness within groups above which a trait
            is considered 'below' the grouping level.

        Returns
        -------
        list of str
            Traits classified as 'above' or 'below' relative to the grouping level.
        """
        if isinstance(full_ds, pl.LazyFrame):
            df = full_ds.collect().to_pandas()
        elif isinstance(full_ds, pl.DataFrame):
            df = full_ds.to_pandas()
        elif isinstance(full_ds, pd.DataFrame):
            df = full_ds
        else:
            raise TypeError("Input must be a pandas DataFrame or polars DataFrame/LazyFrame")

        if trait not in df.columns:
            raise ValueError(f"Trait '{trait}' not found in columns")

        grouped = df.groupby(trait)
        results = {}
        cols = df.columns if include_ids else [col for col in df.columns if ('id' not in col) and ('ind' not in col)]
        for col in cols:
            if col in self.traits and col != trait:
                frac_unique = grouped[col].nunique(dropna=False) / grouped.size()
                results[col] = frac_unique.mean()

        if level == "below":
            return [col for col, frac in results.items() if frac > threshold]
        elif level == "above":
            return [col for col, frac in results.items() if frac <= threshold]
        else:
            raise ValueError("level must be 'below' or 'above'")

    def set_var_objs(self, proj_config, proj_dir):
        """Resolve ``self.col_var_obj``/``self.target_var_obj`` from ``self.col_var_id``/``self.target_var_id``.

        For each of the predictor (``col_var_id``) and target
        (``target_var_id``) variables: builds a ``DataVarConfig``, wraps it
        in a ``VarObject``, and — if ``self.surr_var`` names that variable
        (by side marker ``'x'``/``'y'``, by its resolved name, or by its
        surrogate-timeseries name) — marks that ``VarObject`` to use
        surrogate data (``surr_num``/``ts_type='surr'``). Also backfills
        ``self.col_var``/``self.target_var``/``self.proj_dir`` if unset.

        Parameters
        ----------
        proj_config : cedarkit.core.project_config.ProjectConfig
            Project configuration containing both variables' config blocks.
        proj_dir : str or pathlib.Path
            Root directory of the project.
        """
        col_DataVar = DataVarConfig(proj_config, self.col_var_id, proj_dir)
        self.col_var_obj = VarObject(proj_config, proj_dir, data_var_config=col_DataVar)

        if self.surr_var in ('x', self.col_var_obj.var, self.col_var_obj.surr_ts_var):
            self.col_var_obj.surr_num = self.surr_num
            self.col_var_obj.ts_type = 'surr'

        target_DataVar = DataVarConfig(proj_config, self.target_var_id, proj_dir)
        self.target_var_obj = VarObject(proj_config, proj_dir, data_var_config=target_DataVar)

        if self.surr_var in ('y', self.target_var_obj.var, self.target_var_obj.surr_ts_var):
            self.target_var_obj.surr_num = self.surr_num
            self.target_var_obj.ts_type = 'surr'

        if self.col_var is None:
            self.col_var = self.col_var_obj.var
        if self.target_var is None:
            self.target_var = self.target_var_obj.var

        if self.proj_dir is None:
            self.proj_dir = proj_dir


class DataGroup:
    """Manage a group of CCM runs that share some traits, and the output files matching them.

    ``grp_d``'s trait values are split on construction into
    ``static_traits`` (traits with exactly one value) and
    ``nonstatic_traits`` (traits with zero or multiple values, i.e. ones
    that vary across the group). :meth:`get_files` then walks the output
    directory tree to find files matching those traits and builds one
    ``RunConfig`` per matching file (``file_list``); :meth:`pull_output`
    reads and concatenates those files' data into an ``OutputCollection``.

    Attributes
    ----------
    file_list : list[RunConfig]
        One entry per matched output file, populated by :meth:`get_files`.
    grp_d : dict
        The original group-level trait dict passed to ``__init__``.
    static_traits : dict
        Traits from ``grp_d`` with exactly one value (see :func:`get_static`).
    nonstatic_traits : dict
        Traits from ``grp_d`` with zero or multiple values.
    internal_traits : dict
        Traits whose values are determined during file retrieval rather
        than known up front (populated by :meth:`get_files`).
    parent_config : RunConfig
        Group-level ``RunConfig`` built from ``static_traits``.
    output : OutputCollection or None
        Set externally after pulling output; not populated by this class itself.
    tmp_dir : str or pathlib.Path or None
        Temporary directory for intermediate files.
    missing_files : dict
        Files that were expected but not found or didn't match, populated
        by :meth:`get_files`.
    """

    def __init__(self, grp_d, tmp_dir=None):
        """Split ``grp_d``'s traits into static/nonstatic and build ``parent_config``.

        Parameters
        ----------
        grp_d : dict
            Group-level trait values. Each value is classified as static
            (kept as-is, collapsed to its scalar) if it has exactly one
            element under :func:`correct_iterable`, otherwise as nonstatic
            (kept as given, including ``None``).
        tmp_dir : str or pathlib.Path, optional
            Temporary directory for intermediate files. Default is ``None``.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.file_list = []
        self.grp_d = grp_d  # dictionary of group-level traits
        self.static_traits = {}
        self.nonstatic_traits= {}
        for key, value in grp_d.items():
            if value is not None:
                if len(correct_iterable(value)) == 1:
                    self.static_traits[key]= get_static(value)
                else:
                    self.nonstatic_traits[key] = value
            else:
                self.nonstatic_traits[key] = None

        self.internal_traits = {}
        self.parent_config = RunConfig(self.static_traits, tmp_dir=tmp_dir)

        self.output = None
        self.tmp_dir = tmp_dir
        self.missing_files = {}
        # print('Data group tmp dir', self.tmp_dir)

    def get_metadata_as_iterables(self):
        # Mutator: normalizes every value in self.metadata to a list via correct_iterable. No return value.
        # NOTE: self.metadata is never initialized in __init__ or elsewhere in this class — calling
        # this method currently raises AttributeError unless something external has set self.metadata first.
        self.metadata = {key: correct_iterable(value) for key, value in self.metadata.items()}

    def get_files(self, config, output_path, file_name_pattern=None, source='parquet',
                  discovery_fn=None, row_query_fn=None):
        """Find output files matching this group's traits, building one ``RunConfig`` per match.

        Two modes:

        - **SQLite mode** (``discovery_fn``/``row_query_fn`` given): calls
          ``discovery_fn(output_path, self.grp_d)`` to get a list of trait
          dicts (one per run/file combination), builds a ``RunConfig`` for
          each with ``output_format='sqlite'`` and its query/params from
          ``row_query_fn(trait_dict)``.
        - **Parquet mode** (default): derives a directory template and
          filename pattern from ``config`` (via
          ``output.{source}.dir_structure``/``output.{source}.file_format``),
          walks the filesystem under ``output_path`` for matching filenames
          (:func:`extract_from_pattern`), then does one batched Polars scan
          across all filename-matched candidates to confirm each file
          actually contains rows matching this group's traits before
          building its ``RunConfig``.

        In both modes, traits whose discovered values collapse to a single
        value get folded into ``self.static_traits``; traits with multiple
        discovered values go to ``self.nonstatic_traits``.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig
            Project configuration, used (in Parquet mode) to resolve the
            output directory/filename templates.
        output_path : str or pathlib.Path
            Root directory to search for output files.
        file_name_pattern : str, optional
            Override for the filename pattern (Parquet mode only); if
            ``None``, resolved from ``config``.
        source : str, optional
            Output format key used to resolve templates from ``config`` in
            Parquet mode. Default is ``'parquet'``.
        discovery_fn : callable, optional
            ``discovery_fn(output_path, grp_d) -> list[dict]``. If given,
            switches to SQLite mode.
        row_query_fn : callable, optional
            ``row_query_fn(trait_dict) -> (sql_str, params_dict)``. Required
            in SQLite mode.

        Populates
        ---------
        self.file_list : list[RunConfig]
            One per matched file.
        self.internal_traits : dict
            Traits determined during file retrieval rather than known up front.
        self.missing_files : dict
            Files that were expected but not found, or that failed to match.
        """
        if discovery_fn is not None:
            combinations = discovery_fn(output_path, self.grp_d)
            log_line(self.log, ['get_files: sqlite combinations discovered', len(combinations)],
                     indent=0, log_type="info")
            file_list = []
            nonstatic_updates = defaultdict(set)
            for trait_dict in combinations:
                new_config = self.parent_config.copy()
                new_config.populate(trait_dict)
                new_config.output_path = [output_path]
                new_config.output_format = 'sqlite'
                new_config.output_query, new_config.output_params = row_query_fn(trait_dict)
                file_list.append(new_config)
                for key in new_config.traits:
                    for val in (correct_iterable(getattr(new_config, key)) or []):
                        nonstatic_updates[key].add(val)
            nonstatic_updates = {k: list(v) for k, v in nonstatic_updates.items()}
            for key, values in nonstatic_updates.items():
                if len(values) == 1:
                    self.static_traits[key] = values[0]
                else:
                    self.nonstatic_traits[key] = values
            self.file_list = file_list
            log_line(self.log, ['get_files: sqlite file_list populated', len(file_list)],
                     indent=0, log_type="info")
            return

        grp_path_template = config.get_dynamic_attr("output.{var}.dir_structure", source)
        if file_name_pattern is None:
            file_name_pattern = config.get_dynamic_attr("output.{var}.file_format", source)

        grp_path_template_filled, replaced_parts = template_replace(grp_path_template, self.static_traits)
        log_line(self.log, ['DataGroup get_files: grp_path_template_filled:', grp_path_template_filled], indent=0, log_type="debug")

        known_sections = grp_path_template_filled.split('/')
        bracket_locations = [ik for ik, section in enumerate(known_sections) if '{' in section]
        if len(bracket_locations) > 0:
            _dir_known_section = '/'.join(known_sections[:bracket_locations[0]])
        else:
            _dir_known_section = '/'.join(known_sections)

        self.internal_traits = {key: value for key, value in self.static_traits.items() if key not in replaced_parts}
        for key in self.parent_config.traits:
            if key not in self.static_traits and key not in self.nonstatic_traits:
                self.internal_traits[key] = None

        merged_unaccounted_d = {**self.internal_traits, **self.nonstatic_traits}

        # --- Phase 1: filename-only discovery, no file I/O ---
        candidates = []  # (file_path, file_dict, new_config)
        nonstatic_updates = defaultdict(set)

        for dirpath, _, filenames in os.walk(output_path / _dir_known_section):
            file_dir = Path(dirpath)
            if not filenames:
                continue
            filtered_files = [
                file_dir / fn for fn in filenames
                if (f'.{source}' in fn)
                and 'registry' not in fn
                and fn != 'results.parquet'
                and '.md' not in fn
                and '.yaml' not in fn
                and '.ipynb' not in fn
                and '.png' not in fn
            ]
            for file_path in filtered_files:
                log_line(self.log, [f'DataGroup get_files: checking file {file_path}'], indent=0, log_type="debug")
                try:
                    file_traits = extract_from_pattern(file_path.name, file_name_pattern)
                    file_dict = {**{key: self.static_traits[key] for key in replaced_parts}, **file_traits}

                    fail = False
                    for trait_key in merged_unaccounted_d:
                        if not fail and trait_key in file_dict:
                            if (merged_unaccounted_d[trait_key] is not None
                                    and file_dict[trait_key] not in correct_iterable(merged_unaccounted_d[trait_key])):
                                fail = True
                            else:
                                nonstatic_updates[trait_key].add(file_dict[trait_key])

                    if not fail:
                        new_config = self.parent_config.copy()
                        for key in self.nonstatic_traits:
                            if key not in file_dict or file_dict[key] is None:
                                file_dict[key] = self.nonstatic_traits[key]
                        new_config.populate(file_dict)
                        candidates.append((file_path, file_dict, new_config))

                except ValueError as e:
                    log_line(self.log, e, indent=0, log_type="error")

        log_line(self.log, [f'get_files: filename candidates {len(candidates)}'], indent=0, log_type="debug")

        file_list = []
        missing_files = {}

        if candidates:
            # --- Phase 2: one batch Polars scan across all candidates ---
            candidate_paths = [str(fp) for fp, _, _ in candidates]

            first_schema = pl.scan_parquet(candidate_paths[0]).collect_schema()
            schema_names = set(first_schema.names())

            trait_cols = [c for c in self.parent_config.traits if c in schema_names]

            filter_exprs = [
                pl.col(k).is_in(correct_iterable(v))
                for k, v in self.grp_d.items()
                if v is not None and k in schema_names
            ]

            lf = pl.scan_parquet(candidate_paths, include_file_paths="_source")
            if not trait_cols:
                raise ValueError(
                    f"get_files: none of parent_config traits found in parquet schema {schema_names}. "
                    "Cannot limit column reads."
                )
            lf = lf.select(trait_cols + ["_source"])

            if not filter_exprs:
                raise ValueError(
                    f"get_files: no filter expressions could be built from grp_d {self.grp_d} "
                    "against schema. Refusing unfiltered scan."
                )
            lf = lf.filter(reduce(operator.and_, filter_exprs))

            result = lf.collect()


            # --- Phase 3: build RunConfigs from collected result ---
            if result is not None:
                found_sources = set(result["_source"].to_list())

                for file_path, file_dict, new_config in candidates:
                    src = str(file_path)
                    if src not in found_sources:
                        log_line(self.log, ['get_files: no matching rows for{file_path}'], indent=0, log_type="error")
                        missing_files[file_path] = new_config
                        continue

                    file_rows = result.filter(pl.col("_source") == src)

                    # replicate _internal_query_v1 grp_info logic
                    grp_info = {}
                    for key in self.parent_config.traits:
                        if key in file_rows.columns:
                            grp_info[key] = file_rows[key].unique().to_list()

                    for key, value in new_config.to_dict().items():
                        if value is None:
                            continue
                        grp_info[key] = list(
                            set(correct_iterable(value)) | set(grp_info.get(key, []))
                        )

                    try:
                        file_group_config = RunConfig(grp_info, tmp_dir=self.tmp_dir)
                    except Exception as e:
                        log_line(self.log, ['get_files: RunConfig build failed for', file_path, e], indent=0, log_type="error")
                        missing_files[file_path] = new_config
                        continue

                    file_group_config.output_path = [file_path]
                    log_line(self.log, ['get_files: matched file', file_path], indent=0, log_type="info")
                    file_list.append(file_group_config)

                    for key in file_group_config.traits:
                        for val in (correct_iterable(getattr(file_group_config, key)) or []):
                            nonstatic_updates[key].add(val)

        nonstatic_updates = {key: list(value) for key, value in nonstatic_updates.items()}
        for key, values in nonstatic_updates.items():
            if len(values) == 1:
                self.static_traits[key] = values[0]
            else:
                self.nonstatic_traits[key] = values

        self.file_list = file_list
        log_line(self.log, ['DataGroup get_files: found', len(self.file_list), 'files'], indent=0, log_type="info")
        self.missing_files.update(missing_files)

    def pull_output(self, summary=True, full=False):
        """Read and concatenate all of ``self.file_list``'s output files into one ``OutputCollection``.

        Memory-bounded: each file is scanned, filtered to its own
        ``RunConfig.to_dict()`` trait values, collected, and appended one at
        a time (rather than holding all files' lazy frames open at once).
        Files with no rows after filtering, or with no ``output_path`` set,
        are skipped.

        Parameters
        ----------
        summary : bool, optional
            Currently unused by this method's body.
        full : bool, optional
            Currently unused by this method's body.

        Returns
        -------
        OutputCollection
            Wraps the concatenated tables, with ``grp_specs`` set from
            :meth:`get_group_config`.
        """
        tables = []
        for groupconfig_file in self.file_list:
            if not groupconfig_file.output_path:
                continue
            file_path = groupconfig_file.resolve_output_path(groupconfig_file.output_path[0])
            groupconfig_file.output_path[0] = file_path
            all_traits = groupconfig_file.to_dict()

            lf = pl.scan_parquet(str(file_path))
            schema_names = set(lf.collect_schema().names())
            filter_exprs = [
                pl.col(k).is_in(correct_iterable(v))
                for k, v in all_traits.items()
                if v is not None and k in schema_names
            ]
            if filter_exprs:
                lf = lf.filter(reduce(operator.and_, filter_exprs))
            tbl = lf.collect().to_arrow()
            log_line(self.log, ['pulled table rows', tbl.num_rows], indent=0, log_type="info")
            if tbl.num_rows > 0:
                tables.append(tbl)

        return OutputCollection(grp_specs=self.get_group_config(), in_table=tables, tmp_dir=self.tmp_dir)

    def get_group_config(self):
        # Returns a RunConfig built from the union of static_traits, nonstatic_traits, and internal_traits.
        return RunConfig({**self.static_traits, **self.nonstatic_traits, **self.internal_traits})

#########################################

class Output:
    """A lazily-loaded table of CCM output data, backed by a Parquet file or a SQLite query.

    Wraps a single output table (``_full``) that may be supplied directly,
    or loaded on first access (:meth:`get_table`) from ``path`` — either as
    a Parquet file (default) or via a SQLite query (``format='sqlite'``,
    using ``query``/``params``). ``surrogate``/``real`` split the table on
    the ``surr_var`` column, if present.

    Attributes
    ----------
    _full : polars.LazyFrame or None
        The output table, once loaded. ``None`` until :meth:`get_table` is
        called (directly, or via the ``table``/``full``/``surrogate``/``real``
        properties).
    path : pathlib.Path or None
        File path the table is loaded from/written to.
    type : str or None
        Label for what kind of output this is (e.g. ``'delta_rho_stats'``,
        ``'libsize_aggregated'``).
    tmp_dir : pathlib.Path or None
        Directory used by :meth:`write_table` when ``path`` is unset.
    query : str or None
        SQL query, required when ``format='sqlite'``.
    format : str
        ``'parquet'`` or ``'sqlite'``. Default is ``'parquet'``.
    params : Any or None
        Parameters passed to ``pandas.read_sql_query`` alongside ``query``.
    """

    def __init__(self, full, path=None, outtype=None, tmp_dir=None, query=None, format='parquet', params=None,
                 dyad_dir=None):
        """Wrap an already-loaded table, or set up to lazily load one later.

        Parameters
        ----------
        full : pandas.DataFrame or polars.LazyFrame or polars.DataFrame or None
            The table data, if already available. A pandas DataFrame is
            converted to a polars ``LazyFrame`` immediately; other types are
            stored as-is. If ``None``, the table is loaded on first access
            via :meth:`get_table`.
        path : str or pathlib.Path, optional
            File path to load from / write to.
        outtype : str, optional
            Stored as ``self.type``.
        tmp_dir : str or pathlib.Path, optional
            Directory used by :meth:`write_table` if ``path`` is unset.
        query : str, optional
            SQL query, required when ``format='sqlite'``.
        format : str, optional
            ``'parquet'`` or ``'sqlite'``. Default is ``'parquet'``.
        params : Any, optional
            Parameters passed to ``pandas.read_sql_query`` alongside ``query``.
        dyad_dir : str or pathlib.Path, optional
            Local dyad directory used to rebase a missing path copied from
            another machine.  If omitted, ``tmp_dir`` is used when it follows
            the conventional ``<dyad>/tmp`` layout.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if type(full) is pd.DataFrame:
            full = pl.from_pandas(full).lazy()
        self._full = full
        self.path = path
        self.type = outtype
        self.tmp_dir = tmp_dir
        # Kept separately from ``path`` so serialized objects can retain an
        # old absolute path yet be rebound to the local dyad at read time.
        self.dyad_dir = dyad_dir
        self.query = query
        self.format = format
        self.params = params
        self.ensure_relation_columns()

    def ensure_relation_columns(self):
        """Derive ``relation_spec`` using CedarKit's established surrogate rules.

        ``relation`` remains the generic source/data-file field.  This is the
        non-mutating successor to ``add_relation_s_inferred``: it resolves the
        two variable names from metadata when available, otherwise from the
        relation strings, and adds or replaces ``relation_spec`` without
        creating a semantic ``relation_0`` column.
        """
        if self._full is None:
            return self

        if isinstance(self._full, pd.DataFrame):
            self._full = pl.from_pandas(self._full).lazy()
        elif isinstance(self._full, pl.DataFrame):
            self._full = self._full.lazy()
        if not isinstance(self._full, pl.LazyFrame):
            return self

        schema_names = set(self._full.collect_schema().names())
        if "relation" not in schema_names:
            return self

        if "surr_var" not in schema_names:
            self._full = self._full.with_columns(pl.col("relation").cast(pl.String).alias("relation_spec"))
            return self

        def static_column_value(column):
            if column not in schema_names:
                return None
            values = self._full.select(
                pl.col(column).drop_nulls().cast(pl.String).filter(pl.col(column) != "").unique()
            ).collect()[column]
            return values[0] if len(values) == 1 else None

        x_var_name = static_column_value("x_var")
        y_var_name = static_column_value("y_var")
        if x_var_name is None or y_var_name is None:
            relations = self._full.select(pl.col("relation").drop_nulls().unique()).collect()["relation"].to_list()
            x_var_name, y_var_name = infer_relation_variables(relations)

        relation = pl.col("relation").cast(pl.String)
        surr_var = pl.col("surr_var").cast(pl.String)
        relation_spec = (
            pl.when(relation.str.contains(r"\(surr\)"))
            .then(relation)
            .when(surr_var.is_null() | (surr_var == "") | (surr_var == "neither"))
            .then(relation)
            .when(surr_var == "both")
            .then(
                relation
                .str.replace_all(x_var_name, f"{x_var_name} (surr) ")
                .str.replace_all(y_var_name, f"{y_var_name} (surr) ")
            )
            .when(surr_var.is_in(["x", x_var_name]))
            .then(relation.str.replace_all(x_var_name, f"{x_var_name} (surr) "))
            .when(surr_var.is_in(["y", y_var_name]))
            .then(relation.str.replace_all(y_var_name, f"{y_var_name} (surr) "))
            .otherwise(relation)
            .str.replace_all("  ", " ")
            .str.strip_chars()
        )
        self._full = self._full.with_columns(
            relation_spec.alias("relation_spec")
        )
        return self

    @property
    def surrogate(self):
        """Rows where ``surr_var != 'neither'`` (surrogate-data rows), loading the table if needed.

        Returns
        -------
        polars.LazyFrame or None
            ``None`` if the table has no ``surr_var`` column (i.e. there's
            no way to distinguish surrogate rows).
        """
        self.get_table()
        if 'surr_var' in self._full.collect_schema().names():
            surr_table = self._full.filter(pl.col("surr_var") != "neither")
            return surr_table
        else:
            return None

    @property
    def real(self):
        """Rows where ``surr_var == 'neither'`` (real-data rows), loading the table if needed.

        Returns
        -------
        polars.LazyFrame
            The full table unfiltered if there's no ``surr_var`` column to
            filter on (unlike :attr:`surrogate`, which returns ``None`` in
            that case).
        """
        self.get_table()
        if 'surr_var' in self._full.collect_schema().names():
            real_df = self._full.filter(pl.col("surr_var") == "neither")#.collect()
        else:
            real_df = self._full#.collect()
        return real_df

    @property
    def table(self):
        # Loads the table if needed and returns self._full. Identical to the `full` property below.
        self.get_table()
        return self._full

    @property
    def full(self):
        # Loads the table if needed and returns self._full. Identical to the `table` property above.
        self.get_table()
        # return self._full
        return self._full

    def get_table(self, format=None):
        """Load ``self._full`` from ``self.path`` if it isn't already loaded.

        No-op if ``self._full`` is already set. Otherwise loads according to
        ``format`` (falling back to ``self.format``, then ``'parquet'``):
        a lazy Parquet scan (``format='parquet'``), or a SQLite query read
        via ``pandas.read_sql_query`` then converted to a polars LazyFrame
        (``format='sqlite'``).

        Parameters
        ----------
        format : {'parquet', 'sqlite'}, optional
            Overrides ``self.format`` for this call only (does not mutate
            ``self.format``). Default is ``None`` (use ``self.format``).

        Raises
        ------
        ValueError
            If the required ``path`` (both formats) or ``query`` (sqlite
            only) is missing.
        """
        if self._full is None:
            self.resolve_path()
            if format is None:
                stored_format = getattr(self, "format", None)
                format = stored_format if stored_format is not None else 'parquet'
            if format == 'parquet':
                if self.path is None:
                    raise ValueError(
                        f"Output.get_table() cannot load parquet for outtype={self.type!r}: "
                        "both _full and path are None."
                    )
                self._full = pl.scan_parquet(str(self.path))
            elif format == "sqlite":
                if self.path is None:
                    raise ValueError("Output.path is required for format='sqlite'.")
                if self.query is None:
                    raise ValueError("Output.query is required for format='sqlite'.")

                import sqlite3
                with sqlite3.connect(str(self.path)) as con:
                    if self.params is None:
                        df = pd.read_sql_query(self.query, con)
                    else:
                        df = pd.read_sql_query(self.query, con, params=self.params)

                self._full = pl.from_pandas(df).lazy()
                print('loaded from sqlite, type:', type(self._full), file=sys.stdout, flush=True)
        self.ensure_relation_columns()

    def resolve_path(self, dyad_dir=None, *, prefer_local=False):
        """Rebase this path beneath a local dyad directory, if possible.

        By default existing paths are retained.  With ``prefer_local=True``,
        an existing counterpart below the local dyad takes precedence.  In
        both cases the suffix beneath the dyad name is preserved exactly.
        """
        if dyad_dir is None:
            dyad_dir = getattr(self, 'dyad_dir', None)
        if dyad_dir is None:
            dyad_dir = _dyad_dir_from_tmp(self.tmp_dir)
        resolved = resolve_dyad_anchored_path(self.path, dyad_dir, prefer_local=prefer_local)
        if resolved is not None and resolved != self.path:
            self.path = resolved
        return self.path


    def clear_table(self):
        """Release memory held by this output table."""
        if self._full is not None:
            self._full = None
        gc.collect()

    def write_table(self, tag=''):
        """Write ``self._full`` to a Parquet file at ``self.path``, generating a path if unset.

        If ``self.path`` is ``None``, a scratch path is generated under
        ``self.tmp_dir`` (defaulting to ``./tmp``) using a random UUID,
        prefixed with ``tag`` (or ``self.type``, or ``'scratch'`` if both
        are empty). If ``self.path`` is already set and doesn't contain
        ``'__'``, an attempt is made to prefix the filename stem with
        ``tag`` too — see the notes doc for a caveat on this branch's
        current behavior.

        Parameters
        ----------
        tag : str, optional
            Label used in the generated/prefixed filename. Default is ``''``.

        Raises
        ------
        TypeError
            If ``self._full`` isn't one of the supported table types
            (``polars.LazyFrame``, ``polars.DataFrame``, ``pandas.DataFrame``).
        """
        if tag == '':
            tag = self.type if self.type is not None else 'scratch'
        if self.tmp_dir is None:
            self.tmp_dir = Path(os.getcwd()) / 'tmp'

        if self.path is None:
            unique_scratch_id = uuid.uuid4().hex
            unique_scratch_id = f'{tag}__{unique_scratch_id}'
            scratch_path = self.tmp_dir / f'{unique_scratch_id}.parquet'
            self.path = scratch_path

        if '__' not in str(self.path) and len(tag) >0:
            name = self.path.stem
            self.path = str(self.path).replace('name', f'{tag}__{name}')
        if isinstance(self._full, pl.LazyFrame):
            self._full.collect().write_parquet(self.path)
        elif isinstance(self._full, pl.DataFrame):
            self._full.write_parquet(self.path)
        elif isinstance(self._full, pd.DataFrame):
            pl.from_pandas(self._full).write_parquet(self.path)
        else:
            raise TypeError(f"Unsupported table type for write_table: {type(self._full)}")


class OutputCollection:
    """A group's CCM output tables, plus derived metrics and the relationship they describe.

    Wraps several lazily-managed :class:`Output` tables (raw ``table``,
    ``libsize_aggregated``, ``active_stats``/``active_full``,
    ``delta_rho_stats``/``delta_rho_full``) produced by running CCM over a
    group of runs (``grp_config``), and a :class:`Relationship` (split into
    ``r1``/``r2`` sides) describing what those tables say about the two
    variables involved. The ``calc_*``/``aggregate_*`` methods populate the
    ``Output`` attributes from raw data; :meth:`calc_metrics` (and the
    lag-finding pipeline it drives — see that method's docstring) populates
    ``r1``/``r2``'s scalar summary attributes from
    ``delta_rho_stats``/``delta_rho_full``.

    ``_COMPAT_DEFAULTS`` and the `from_legacy`/`_ensure_compat_attributes`/
    `__setstate__` machinery exist to backfill attributes on instances that
    were pickled before some of these attributes were introduced.

    Attributes
    ----------
    dyad_home : pathlib.Path or None
        Home directory for this variable pair's ("dyad's") analysis.
    tmp_path : pathlib.Path or None
        Temporary directory for intermediate files.
    grp_config : RunConfig or None
        Group-level configuration this collection's data came from.
    label_stem : str or None
        Label stem for output files.
    table : Output or None
        The raw/combined CCM output table.
    libsize_aggregated : Output or None
        Output aggregated across libsize, from :meth:`aggregate_libsize`.
    active_stats : Output or None
        Reserved for active-statistics output (not currently populated by
        any method on this class).
    active_full : Output or None
        Reserved for active-full output (not currently populated by any
        method on this class).
    delta_rho_stats : Output or None
        Delta-rho summary statistics, from :meth:`calc_delta_rho`.
    delta_rho_full : Output or None
        Delta-rho full bootstrap vectors, from :meth:`calc_delta_rho`.
    relationships : Relationship or None
        Describes the x/y relationship for ``grp_config``.
    r1 : RelationshipSide or None
        First side of ``relationships``.
    r2 : RelationshipSide or None
        Second side of ``relationships``.
    """

    _COMPAT_DEFAULTS = {
        'dyad_home': None,
        'grp_config': None,
        'label_stem': None,
        'table': None,
        'libsize_aggregated': None,
        'active_stats': None,
        'active_full': None,
        'delta_rho_stats': None,
        'delta_rho_full': None,
        'lag_choices': None,
        'relationships': None,
        'r1': None,
        'r2': None,
        'real_r_df_tmp': None,
        'surr_r_df_tmp': None,
        'tmp_path': None,
    }

    def __init__(self, grp_specs=None, in_table=None, outtype=None, tmp_dir=None):
        """Build ``grp_config``/``relationships`` from ``grp_specs``, and ``table`` from ``in_table``.

        ``grp_specs`` becomes ``self.grp_config`` directly if it's already a
        ``RunConfig``, or is wrapped in one if it's a dict. A GraphCM
        ``RunConfig`` is retained directly when it exposes ``get_trait_value``
        and its directed ``relationship``. ``in_table`` is normalized to a
        list and handled one of two ways: if every element is an
        ``OutputCollection``, their corresponding ``Output`` attributes are
        merged in via :meth:`combine_OutputCollections`; otherwise each
        element (a ``polars.DataFrame``/``LazyFrame``, ``pandas.DataFrame``,
        or ``Output``) is converted to a polars
        ``LazyFrame`` and concatenated into ``self.table``.

        Parameters
        ----------
        grp_specs : RunConfig, GraphCM RunConfig, or dict, optional
            Group-level configuration, or trait dict to build a CedarKit
            ``RunConfig`` from.
        in_table : Any or list[Any], optional
            Table(s) (or other ``OutputCollection`` instances) to populate
            ``self.table`` from.
        outtype : str, optional
            Passed through to the resulting ``Output``'s ``type``.
        tmp_dir : str or pathlib.Path, optional
            Temporary directory. Defaults to ``grp_config.proj_dir / 'tmp'``
            if available, else ``Path.cwd() / 'tmp'``. Created if it doesn't
            exist.

        Notes
        -----
        ``self.relationships`` is computed twice: once near the top of this
        method (while ``self.grp_config`` is still ``None``, so this first
        computation is always ``None``) and again at the end (after
        ``self.grp_config`` has actually been set from ``grp_specs``, so
        this second computation is the one that sticks). ``self.r1``/
        ``self.r2``, however, are only computed once, during that first
        (always-``None``-relationship) pass — there is no corresponding
        recomputation of ``r1``/``r2`` at the end of ``__init__`` the way
        there is for ``relationships``. On a quick read this looks like it
        would leave ``self.r1``/``self.r2`` as ``None`` after construction
        whenever ``grp_specs`` is given, even though ``self.relationships``
        ends up correctly populated — worth verifying directly (e.g. by
        constructing an instance and checking ``r1``/``r2``) rather than
        assuming, since :meth:`get_relationship`/:meth:`set_relationships`
        or :meth:`_ensure_compat_attributes` may be relied upon elsewhere in
        the calling code to populate them before they're used, and there may
        be context from how this class is used downstream that isn't
        visible from this file alone.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.dyad_home = None

        self.grp_config = None
        self.label_stem = None

        # print('created temporary directory', self.tmp_path)
        # output = self.prep_output(in_table, use_case='full')
        self.table = None #Output(in_table) if in_table is not None else None

        self.libsize_aggregated = None
        self.active_stats = None
        self.active_full = None
        self.delta_rho_stats = None
        self.delta_rho_full = None
        self.lag_choices = None
        self.viable_lags = None

        self.relationships =  Relationship(self.grp_config.var_x, self.grp_config.var_y) if self.grp_config is not None else None
        self.r1 = RelationshipSide('r1', relationship=self.relationships) if self.relationships is not None else None
        self.r2 = RelationshipSide('r2', relationship=self.relationships) if self.relationships is not None else None

        self.real_r_df_tmp = None
        self.surr_r_df_tmp = None

        if isinstance(grp_specs, RunConfig):
            self.grp_config = grp_specs
        elif isinstance(grp_specs, dict):
            iterable_d = {k: correct_iterable(v) for k, v in grp_specs.items()}
            self.grp_config = RunConfig(iterable_d)
        elif (
            hasattr(grp_specs, "get_trait_value")
            and hasattr(grp_specs, "to_param_d")
            and hasattr(grp_specs, "relationship")
        ):
            self.grp_config = grp_specs

        config_proj_dir = getattr(self.grp_config, "proj_dir", None)
        self.tmp_path = tmp_dir if tmp_dir is not None else (config_proj_dir / 'tmp' if config_proj_dir is not None else Path.cwd() / 'tmp')

        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.dyad_home = None
        # print('temporary directory for OutputCollection:', self.tmp_path)


    # def __init__(self, in_table):
        # Local helper: coerces obj to a polars LazyFrame (or None). A near-duplicate
        # of the _to_lazy_frame closure in combine_OutputCollections below; this one
        # additionally backfills missing trait columns onto a pandas DataFrame from
        # self.grp_config before converting it. See notes_core_todos.md (utility-extraction candidates).
        def _to_lazy_frame(obj):
            if obj is None:
                return None
            if isinstance(obj, pl.LazyFrame):
                return obj
            if isinstance(obj, pl.DataFrame):
                return obj.lazy()
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
                if isinstance(self.grp_config, RunConfig):
                    for col in [
                        'E', 'tau', 'Tp', 'lag', 'knn', 'surr_var', 'surr_num',
                        'x_id', 'x_age_model_ind', 'x_var', 'y_id',
                        'y_age_model_ind', 'y_var', 'LibSize', 'ind_i',
                        'relation', 'forcing', 'responding',
                    ]:
                        if col not in df.columns:
                            df[col] = self.grp_config.get_trait_value(col)
                return pl.from_pandas(df).lazy()
            return None

        if isinstance(in_table, list) is False:
            in_table = [in_table]

        if isinstance(in_table, list) and len(in_table) > 0 and isinstance(in_table[0], OutputCollection):
            outputcollections = [outputcoll for outputcoll in in_table if (outputcoll is not None) and (isinstance(outputcoll, OutputCollection) is True)]
            for attr in ['table', 'libsize_aggregated', 'active_stats', 'active_full', 'delta_rho_stats', 'delta_rho_full']:
                try:
                    self.combine_OutputCollections(attr, outputcollections)
                except Exception as e:
                    log_line(self.log, f'Error combining OutputCollections for attribute {attr}: {e}',
                             indent=0, log_type="error")
                    # print(f'Error combining OutputCollections for attribute {attr}: {e}')
        elif isinstance(in_table, list):
            lazy_tables = []
            for tbl in in_table:
                if isinstance(tbl, Output):
                    if tbl._full is None and tbl.path is None:
                        lazy_tbl = None
                    else:
                        lazy_tbl = _to_lazy_frame(tbl.table)
                else:
                    lazy_tbl = _to_lazy_frame(tbl)
                if lazy_tbl is not None:
                    lazy_tables.append(lazy_tbl)

            if len(lazy_tables) > 0:
                if len(lazy_tables) == 1:
                    combined = lazy_tables[0]
                else:
                    combined = pl.concat(lazy_tables, how='diagonal_relaxed')
                self.table = Output(combined, outtype=outtype, tmp_dir=self.tmp_path)

        if isinstance(self.grp_config, RunConfig):
            self.relationships = Relationship(self.grp_config.var_x, self.grp_config.var_y)
        elif self.grp_config is not None:
            self.set_relationships(relationship=self.grp_config.relationship)

    @classmethod
    def from_legacy(cls, legacy_obj, grp_specs=None, tmp_dir=None):
        """Upgrade a previously serialized ``OutputCollection`` to have all current attributes.

        If ``legacy_obj`` already has every key in ``_COMPAT_DEFAULTS``, it's
        backfilled in place via :meth:`_ensure_compat_attributes` and
        returned as-is. Otherwise, a fresh instance is constructed (using
        ``grp_specs``/``tmp_dir``, or ``legacy_obj``'s own ``grp_config``/
        ``tmp_path`` if not given) and every attribute ``legacy_obj`` already
        has is copied onto it, before backfilling any still-missing ones.

        Parameters
        ----------
        legacy_obj : OutputCollection
            Object to upgrade.
        grp_specs : RunConfig or dict, optional
            Overrides ``legacy_obj.grp_config`` for the fresh instance, if given.
        tmp_dir : str or pathlib.Path, optional
            Overrides ``legacy_obj.tmp_path`` for the fresh instance, if given.

        Returns
        -------
        OutputCollection

        Raises
        ------
        ValueError
            If ``legacy_obj`` is ``None``.
        """
        if legacy_obj is None:
            raise ValueError("legacy_obj cannot be None")
        if isinstance(legacy_obj, cls) and all(hasattr(legacy_obj, attr) for attr in cls._COMPAT_DEFAULTS):
            legacy_obj._ensure_compat_attributes()
            return legacy_obj

        resolved_grp_specs = grp_specs if grp_specs is not None else getattr(legacy_obj, 'grp_config', None)
        resolved_tmp_dir = tmp_dir if tmp_dir is not None else getattr(legacy_obj, 'tmp_path', None)

        upgraded = cls(grp_specs=resolved_grp_specs, in_table=[], tmp_dir=resolved_tmp_dir)

        for attr in cls._COMPAT_DEFAULTS:
            if hasattr(legacy_obj, attr):
                setattr(upgraded, attr, getattr(legacy_obj, attr))

        upgraded._ensure_compat_attributes()
        return upgraded

    def _ensure_compat_attributes(self):
        """Backfill any missing ``_COMPAT_DEFAULTS`` attributes, and renormalize derived state.

        Sets any attribute named in ``_COMPAT_DEFAULTS`` that this instance
        doesn't already have to its default value, then: normalizes
        ``tmp_path`` to a ``Path`` (creating it if needed, deriving a default
        from ``grp_config.proj_dir`` or the cwd if unset), rebuilds
        ``relationships``/``r1``/``r2`` from ``grp_config`` if any of them
        are ``None``, and propagates ``tmp_path`` onto every populated
        ``Output`` attribute's ``tmp_dir``.
        """
        for attr, default in self._COMPAT_DEFAULTS.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)

        # Normalize tmp_path on legacy objects that may not have had it.
        if self.tmp_path is None:
            if self.grp_config is not None and getattr(self.grp_config, 'proj_dir', None) is not None:
                self.tmp_path = self.grp_config.proj_dir / 'tmp'
            else:
                self.tmp_path = Path.cwd() / 'tmp'

        self.tmp_path = Path(self.tmp_path)
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        if self.relationships is None and self.grp_config is not None:
            self.relationships = Relationship(self.grp_config.var_x, self.grp_config.var_y)

        if isinstance(self.relationships, Relationship):
            if self.r1 is None:
                self.r1 = RelationshipSide('r1', relationship=self.relationships)
            if self.r2 is None:
                self.r2 = RelationshipSide('r2', relationship=self.relationships)
        elif self.relationships is not None and self.r1 is None:
            # A supplied directed relationship is itself the only active side.
            self.r1 = self.relationships

        for out_attr in ['table', 'libsize_aggregated', 'active_stats', 'active_full', 'delta_rho_stats', 'delta_rho_full']:
            out = getattr(self, out_attr)
            if out is not None and hasattr(out, 'tmp_dir'):
                out.tmp_dir = self.tmp_path

    def __setstate__(self, state):
        # Unpickling hook: restores __dict__, rebuilds the logger (not picklable), then
        # backfills any attributes missing from an older pickle via _ensure_compat_attributes.
        self.__dict__.update(state)
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._ensure_compat_attributes()

    # note: the default output_convention was 'influence' prior to may 29, 2026
    def set_relationships(self, influence_word="causes", operation_word="reconstructs", output_convention="operation", pres_convention="influence",
        convention_mapping=None, relationship=None):
        """Set an explicit directed relationship or build CedarKit's two-sided one.

        With ``relationship=None``, this retains CedarKit's existing behavior:
        construct a bidirectional :class:`Relationship` from ``self.grp_config``
        and its ``r1``/``r2`` sides.  With a supplied directed relationship,
        ``relationships`` and ``r1`` reference that object and ``r2`` is
        ``None``.  A supplied relationship must expose ``r`` and ``r_calc``;
        metric calculation additionally uses ``participant_variables``.

        Parameters
        ----------
        influence_word : str, optional
            Verb for "causes" in pres-convention sentences. Default is ``"causes"``.
        operation_word : str, optional
            Verb for "reconstructs" in calc-convention sentences. Default is ``"reconstructs"``.
        output_convention : {'operation', 'influence'}, optional
            Calc-output sentence form. Default is ``"operation"``.
        pres_convention : {'operation', 'influence'}, optional
            Presentation-output sentence form. Default is ``"influence"``.
        convention_mapping : dict, optional
            Optional word remapping. Default is ``None``.
        relationship : object, optional
            Explicit directed relationship, such as GraphCCM's
            ``Relationship``. Default is ``None``.
        """
        if relationship is not None:
            if not hasattr(relationship, "r") or not hasattr(relationship, "r_calc"):
                raise TypeError("A supplied relationship must expose r and r_calc.")
            self.relationships = relationship
            self.r1 = relationship
            self.r2 = None
            return
        self.relationships = Relationship(self.grp_config.var_x, self.grp_config.var_y,
                                          operation_word=operation_word, output_convention=output_convention,
                                          pres_convention=pres_convention, convention_mapping=convention_mapping) if self.grp_config is not None else None
        self.r1 = RelationshipSide('r1', relationship=self.relationships, operation_word=operation_word, output_convention=output_convention,
                                          pres_convention=pres_convention, convention_mapping=convention_mapping) if self.relationships is not None else None
        self.r2 = RelationshipSide('r2', relationship=self.relationships, operation_word=operation_word, output_convention=output_convention,
                                          pres_convention=pres_convention, convention_mapping=convention_mapping) if self.relationships is not None else None

    def get_relationship(self, relationship_id='r1', output_convention="operation"):
        """Return an active relationship side, building CedarKit sides if needed.

        This is the safe way to access ``r1``/``r2`` — unlike reading
        ``self.r1``/``self.r2`` directly, it builds them on demand if
        ``self.relationships`` is currently ``None``.

        Parameters
        ----------
        relationship_id : {'r1', 'r2'}, optional
            Which side to return. A supplied directed relationship exposes only
            ``'r1'``. Default is ``'r1'``.
        output_convention : str, optional
            Passed to :meth:`set_relationships` if it needs to be called.
            Default is ``"operation"``.

        Returns
        -------
        RelationshipSide or directed relationship

        Raises
        ------
        ValueError
            If ``relationship_id`` is unsupported or requests ``'r2'`` from a
            directed relationship.
        """
        if self.relationships is None:
            self.set_relationships(output_convention=output_convention)
        if relationship_id == 'r1':
            return self.r1
        if relationship_id == 'r2':
            if self.r2 is None:
                raise ValueError("A supplied directed relationship has no r2 side.")
            return self.r2
        raise ValueError(f"Unsupported relationship_id '{relationship_id}'. Use 'r1' or 'r2'.")

    def relation_aliases(self, relationship_id):
        """Return generic relation spellings that select one relationship category.

        CedarKit's :class:`Relationship` owns the normal r1/r2 vocabulary.
        The fallbacks retain support for an externally supplied directed
        relationship object, which need only expose ``r`` and ``r_calc``.
        """
        relationships = self.relationships
        if relationships is None:
            return [relationship_id]

        if relationship_id in {"r1", "r2"} and hasattr(relationships, "relation_aliases"):
            return list(relationships.relation_aliases(relationship_id))

        if relationship_id == "r1" and self.r2 is None:
            aliases = [getattr(relationships, attr, None) for attr in ("r_calc", "r", "formulation")]
            return [alias for alias in aliases if alias is not None and "(surr)" not in alias]

        aliases = [relationship_id]
        for mapping_name in ("to_calc_mapping", "to_pres_mapping"):
            mapping = getattr(relationships, mapping_name, None)
            if isinstance(mapping, dict):
                aliases.append(mapping.get(relationship_id, relationship_id))
        return list(dict.fromkeys(alias for alias in aliases if alias is not None))

    def combine_OutputCollections(self, attr, other_output_collections):
        """Concatenate one named ``Output`` attribute across this and other ``OutputCollection``s.

        Gathers ``getattr(self, attr)`` and ``getattr(oc, attr)`` for each
        ``oc`` in ``other_output_collections``, drops any that are ``None``,
        converts each to a polars ``LazyFrame`` (clearing each source
        ``Output``'s table after conversion to bound memory use), and
        concatenates them into a single new ``Output`` assigned back to
        ``getattr(self, attr)``. No-ops (returns ``self`` unchanged) if
        every source is ``None``.

        Parameters
        ----------
        attr : str
            Name of the ``Output``-typed attribute to combine (e.g.
            ``'table'``, ``'delta_rho_stats'``).
        other_output_collections : OutputCollection or list[OutputCollection]
            Other collection(s) to merge ``attr`` in from.

        Returns
        -------
        OutputCollection
            ``self``, for chaining.
        """
        print('combining OutputCollections for', attr)
        tables = [getattr(self, attr)]
        print('combining', attr)
        if not isinstance(other_output_collections, list):
            other_output_collections = [other_output_collections]

        for other_output_collection in other_output_collections:
             tables.append(getattr(other_output_collection, attr))
        print(len(tables), 'tables to combine for', attr)
        tables = [tbl for tbl in tables if tbl is not None]
        if len(tables) == 0:
            return self

        # Local helper: coerces obj to a polars LazyFrame (or None). Near-duplicate of the
        # _to_lazy_frame closure in __init__ above, minus that one's pandas trait-column backfill.
        def _to_lazy_frame(obj):
            if obj is None:
                return None
            if isinstance(obj, pl.LazyFrame):
                return obj
            if isinstance(obj, pl.DataFrame):
                return obj.lazy()
            if isinstance(obj, pd.DataFrame):
                return pl.from_pandas(obj).lazy()
            return None

        tables_full = []
        outtypes = []
        for tbl in tables:
            if isinstance(tbl, Output):
                if tbl._full is None and tbl.path is None:
                    lazy_tbl = None
                else:
                    lazy_tbl = _to_lazy_frame(tbl.table)
                if lazy_tbl is not None:
                    tables_full.append(lazy_tbl)
                    outtypes.append(tbl.type)
                    tbl.clear_table()
            else:
                lazy_tbl = _to_lazy_frame(tbl)
                if lazy_tbl is not None:
                    tables_full.append(lazy_tbl)

        outtypes = list(set(outtypes))
        outtype = outtypes[0] if len(outtypes) == 1 else attr
        if len(tables_full) > 0:
            combined = tables_full[0] if len(tables_full) == 1 else pl.concat(tables_full, how='diagonal_relaxed')
            setattr(self, attr, Output(combined, outtype=outtype, tmp_dir=self.tmp_path))
            print('combined', attr)

        return self

    def calc_metrics(self, relationship_id=None, lag=None, smoothing_window=1):
        """Pick a target lag and populate ``r1``/``r2``'s summary metrics from ``delta_rho_stats``/``delta_rho_full``.

        Entry point for a multi-step, stateful pipeline that threads through
        several methods below, each reading/writing shared instance state:

        1. :meth:`_calc_metrics` (per relationship side) checks
           ``self.target_lag``; if unset, calls :meth:`set_target_lag`.
        2. :meth:`set_target_lag` ensures ``self.lag_choices`` exists (via
           :meth:`calc_lags_peaks`, which calls :meth:`find_candidate_peaks`
           on ``self.delta_rho_full``'s real-data rows), then calls
           :meth:`find_viable_peaks` (which calls :meth:`lag_is_equivalent`)
           to get ``self.viable_lags``, then :meth:`test_lags` to score those
           candidates against surrogate performance, and finally picks
           ``self.target_lag`` from the result.
        3. Back in :meth:`_calc_metrics`, the row of ``self.viable_lags``
           matching ``self.target_lag`` is used to set scalar attributes
           (``delta_rho``, ``maxlibsize_rho``, ``lag``, surrogate
           outperformance counts/fractions, peak bounds) directly on
           ``self.r1`` or ``self.r2`` — accessed as `self.r1`/`self.r2`
           directly rather than via :meth:`get_relationship`, so this relies
           on them already being populated (see the caveat in
           :meth:`__init__`'s docstring about whether that's guaranteed).

        If ``relationship_id`` is ``None``, CedarKit's two-sided relationship
        processes both sides. A supplied directed relationship processes only
        ``r1``. Each attempted side's failure is caught and printed.

        Parameters
        ----------
        relationship_id : {'r1', 'r2', None}, optional
            Which side to compute metrics for; both if ``None``. Default is ``None``.
        lag : None or callable or {'pos', 'neg'}, optional
            Restricts candidate lags; passed through to
            :meth:`set_target_lag` → :meth:`_resolve_metrics_lag_filter`.
            Default is ``None`` (unrestricted).
        smoothing_window : int, optional
            Smoothing window passed through the pipeline to
            :meth:`find_candidate_peaks`/:meth:`find_viable_peaks`. Default is ``1``.
        """
        self.delta_rho_stats.get_table()
        if relationship_id is None:
            relationship_ids = ('r1',) if self.r2 is None else ('r1', 'r2')
            for active_relationship_id in relationship_ids:
                try:
                    self._calc_metrics(active_relationship_id, lag=lag, smoothing_window=smoothing_window)
                except Exception as e:
                    print(f'Error calculating metrics for {active_relationship_id}: {e}')
        else:
            self._calc_metrics(relationship_id=relationship_id, lag=lag, smoothing_window=smoothing_window)
        self.delta_rho_stats.clear_table()

    @staticmethod
    def _resolve_metrics_lag_filter(lag=None):
        """
        Lag filtering used by calc_metrics/_calc_metrics.
        Supported:
        - None: unrestricted
        - 'pos' / 'neg': legacy string modes
        - callable: custom predicate, e.g. lambda x: x > -3
        """
        if lag is None:
            return lambda _: True
        if callable(lag):
            return lag
        if lag == 'pos':
            return lambda x: x >= 0
        if lag == 'neg':
            return lambda x: x <= 0
        raise TypeError("lag must be None, a callable, or one of the legacy string modes: 'pos'/'neg'")

    # def _resolve_relationship_name(self, relationship_id='r1'):
    #     if self.relationships is None:
    #         self.set_relationships()
    #     if relationship_id == 'r1':
    #         return self.relationships.r1
    #     if relationship_id == 'r2':
    #         return self.relationships.r2
    #     raise ValueError(f"Unsupported relationship_id '{relationship_id}'. Use 'r1' or 'r2'.")
    #
    # def _resolve_relationship_name_calc(self, relationship_id='r1'):
    #     if self.relationships is None:
    #         self.set_relationships()
    #     if relationship_id == 'r1':
    #         return self.relationships.r1_calc
    #     if relationship_id == 'r2':
    #         return self.relationships.r2_calc
    #     raise ValueError(f"Unsupported relationship_id '{relationship_id}'. Use 'r1' or 'r2'.")


    # def _draw_metric_df(self, source, table_attr='real'):
    #
    #
    #     gb = source.group_by(["relation", 'lag', 'surr_var', 'surr_num']).aggregate(
    #         [("maxlibsize_rho", "mean"),
    #          ("maxlibsize_rho", "stddev"),
    #          ("maxlibsize_rho", 'approximate_median'),
    #          ('maxlibsize_rho', 'tdigest', tdigest_opts),
    #          ("delta_rho", "mean"), ("delta_rho", "stddev"),
    #          ('delta_rho', 'approximate_median'),
    #          ('delta_rho', 'tdigest', tdigest_opts),
    #          ])
    #
    #     gb_df = gb.to_pandas()
    #     for var in ['maxlibsize_rho', 'delta_rho']:
    #         for ik, q in enumerate([.25, .75]):
    #             number_label = str(q).replace('.','p').lstrip('0')
    #             gb_df[f'{var}_{number_label}'] = gb_df[f'{var}_tdigest'].apply(lambda x: x[ik] if x is not None else np.nan)
    #     return gb_df

    def _draw_metric_df(self, source, table_attr='real'):
        """Group ``source`` by (relation, lag, surr_var, surr_num) and summarize ``maxlibsize_rho``/``delta_rho``.

        Step used by :meth:`calc_lags_peaks` (see :meth:`calc_metrics` for
        the full pipeline this is part of). Computes mean/std/median/p25/p75
        for both metrics per group.

        Parameters
        ----------
        source : polars.DataFrame or polars.LazyFrame
            Table with ``relation``, ``lag``, ``surr_var``, ``surr_num``,
            ``maxlibsize_rho``, and ``delta_rho`` columns.
        table_attr : str, optional
            Currently unused by this method's body (descriptive only at
            call sites — passed as ``'real'``/``'surrogate'``).

        Returns
        -------
        pandas.DataFrame
        """
        gb = source.group_by(["relation", "lag", "surr_var", "surr_num"]).agg(
            pl.col("maxlibsize_rho").mean().alias("maxlibsize_rho_mean"),
            pl.col("maxlibsize_rho").std().alias("maxlibsize_rho_stddev"),
            pl.col("maxlibsize_rho").median().alias("maxlibsize_rho_approximate_median"),
            pl.col("maxlibsize_rho").quantile(0.25).alias("maxlibsize_rho_p25"),
            pl.col("maxlibsize_rho").quantile(0.75).alias("maxlibsize_rho_p75"),

            pl.col("delta_rho").mean().alias("delta_rho_mean"),
            pl.col("delta_rho").std().alias("delta_rho_stddev"),
            pl.col("delta_rho").median().alias("delta_rho_approximate_median"),
            pl.col("delta_rho").quantile(0.25).alias("delta_rho_p25"),
            pl.col("delta_rho").quantile(0.75).alias("delta_rho_p75"),
        )

        if isinstance(gb, pl.LazyFrame):
            return gb.collect().to_pandas()
        return gb.to_pandas()

    # @TODO: this should reference a metric utility
    def find_candidate_peaks(self, df, y_col='maxlibsize_rho', x_col='lag', smoothing_window=1):
        """Find local maxima of ``{y_col}_mean`` over ``x_col`` (lag) and estimate each peak's plateau bounds.

        Step used by :meth:`calc_lags_peaks` (see :meth:`calc_metrics` for
        the full pipeline). Sorts by ``x_col``, optionally smooths
        ``{y_col}_mean`` with a centered rolling mean, finds sign changes in
        its derivative (positive-to-non-positive, i.e. local maxima) as
        candidate peaks, then for each candidate (ordered by descending
        ``{y_col}_mean``) estimates a plateau of ``x_col`` values around it
        whose ``{y_col}_p25``/``{y_col}_p75`` interquantile range overlaps
        the peak's mean value.

        Parameters
        ----------
        df : pandas.DataFrame
            Must have columns ``x_col``, ``{y_col}_mean``, ``{y_col}_p25``,
            ``{y_col}_p75`` (the output shape of :meth:`_draw_metric_df`).
        y_col : str, optional
            Metric column stem. Default is ``'maxlibsize_rho'``.
        x_col : str, optional
            Column to find peaks over. Default is ``'lag'``.
        smoothing_window : int, optional
            Rolling-mean window applied to ``{y_col}_mean`` before peak
            detection, if greater than 1. Default is ``1`` (no smoothing).

        Returns
        -------
        pandas.DataFrame
            ``df`` rows at candidate peak lags, sorted by descending
            ``{y_col}_mean``, with added ``peak_start``/``peak_end``
            (integer positions in ``x_col``) and
            ``peak_start_deriv``/``peak_end_deriv`` (the surrounding
            zero-crossing bounds) columns.
        """
        # y = df[y_col].values
        y_col_var = f'{y_col}_mean'
        deriv_var = f'd{y_col_var}_d{x_col}'
        df=df.sort_values(by=x_col).reset_index(drop=True)
        if smoothing_window>1:
            df[f'{y_col_var}_smooth'] = df[y_col_var].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        else:
            df[f'{y_col_var}_smooth'] = df[y_col_var]
        deriv_var_source = f'{y_col_var}_smooth'
        df[deriv_var] = df[deriv_var_source] - df[deriv_var_source].shift(1)
        # print(df)

        peak_indices = []
        y = df[[deriv_var, x_col]].copy().values
        # print(df[(df['lag']>-10) & (df['lag']<10)])
        # print('derivative values for peak finding:', y)
        lags = []
        for i in range(1, len(y) - 1):
            if (y[i][0] > 0 and y[i + 1][0] <= 0):
                peak_indices.append(i)
                lags.append(y[i][1])
            # if y[i] > y[i - 1] and y[i] > y[i + 1]:
            #     peak_indices.append(i)

        # might need to add logic to capture plateaus, e.g. y[i] > 0 and y[i + 1] <= 0 and y[i-1] < y[i]
        # print('lag values at candidate peaks:', lags)
        ordered_peaks = df[df['lag'].isin(lags)].copy().sort_values(by=y_col_var, ascending=False).reset_index(drop=True)
        ordered_peaks['peak_start'] = None
        ordered_peaks['peak_end'] = None
        ordered_peaks['peak_start_deriv'] = None
        ordered_peaks['peak_end_deriv'] = None
        for idx in ordered_peaks.index:
            entry = df[df['lag']<=ordered_peaks.loc[idx, x_col]].copy().sort_values(by=x_col, ascending=False)
            lag_window_start = entry[entry[deriv_var]<0][x_col].max()
            exit = df[df['lag'] > ordered_peaks.loc[idx, x_col]].copy().sort_values(by=x_col, ascending=True)
            lag_window_end = exit[exit[deriv_var] > 0][x_col].min()
            ordered_peaks.loc[idx, 'peak_start_deriv'] = lag_window_start
            ordered_peaks.loc[idx, 'peak_end_deriv'] = lag_window_end

            # tighter leash
            peak_lag = ordered_peaks.loc[idx, x_col]
            peak_mean = ordered_peaks.loc[idx, y_col_var]

            lo_col = f'{y_col}_p25'
            hi_col = f'{y_col}_p75'

            arr_x = df[x_col].to_numpy()
            arr_lo = df[lo_col].to_numpy()
            arr_hi = df[hi_col].to_numpy()

            peak_pos = np.where(arr_x == peak_lag)[0]
            if len(peak_pos) == 0:
                # ordered_peaks.loc[idx, "peak_start"] = np.nan
                # ordered_peaks.loc[idx, "peak_end"] = np.nan
                continue
            peak_pos = int(peak_pos[0])

            valid = ~pd.isna(arr_lo) & ~pd.isna(arr_hi) & ~pd.isna(arr_x)
            overlap = valid & (arr_lo <= peak_mean) & (arr_hi >= peak_mean)
            overlap[peak_pos] = True

            left = peak_pos
            while left > 0 and overlap[left - 1]:
                left -= 1

            right = peak_pos
            while right < len(arr_x) - 1 and overlap[right + 1]:
                right += 1

            ordered_peaks.loc[idx, 'peak_start'] = left
            ordered_peaks.loc[idx, 'peak_end'] = right
        # print('ordered candidate peaks:', ordered_peaks.head())

        return ordered_peaks

    def lag_is_equivalent(self, candidates, target=None, variable='maxlibsize_rho', category='', smoothing_window=1):
        """From a set of candidate peaks, keep ``target`` plus any statistically indistinguishable from it.

        Step used by :meth:`find_viable_peaks` (see :meth:`calc_metrics` for
        the full pipeline). Sorts ``candidates`` by descending
        ``{variable}_mean``; if ``target`` isn't given, the top candidate is
        used as ``target`` and excluded from the comparison set. A
        candidate is kept if its IQR overlaps ``target``'s p25 (a somewhat
        arbitrary choice — see the inline ``# TODO`` on this) or if its
        mean +/- stddev band overlaps ``target``'s.

        Parameters
        ----------
        candidates : pandas.DataFrame
            Candidate peak rows (e.g. from :meth:`find_candidate_peaks`).
        target : dict, optional
            Reference peak to compare others against. If ``None``, the
            top-ranked candidate is used.
        variable : str, optional
            Metric column stem. Default is ``'maxlibsize_rho'``.
        category : str, optional
            Label written into the result's ``'category'`` column.
        smoothing_window : int, optional
            If greater than 1, compares on the smoothed variable instead of
            the p25/p75 columns. Default is ``1``.

        Returns
        -------
        pandas.DataFrame
            ``target`` plus equivalent candidates, with a ``'category'``
            column set to ``category``. Returns ``candidates`` unchanged
            (copied) if it has 0 or 1 rows.
        """
        variable_stem = variable
        variable = f'{variable_stem}_mean'
        comp_var_target = f'{variable_stem}_p25'
        comp_var_alt = f'{variable_stem}_p75'
        if smoothing_window>1:
            variable = f'{variable}_smooth'
            comp_var_target = variable
            comp_var_alt = variable

        candidates_list = candidates.sort_values(by=f'{variable}', ascending=False).to_dict(orient='records')
        # top n
        if len(candidates_list)>1:
            if target is None:
                target = candidates_list.pop(0)
            top_candidates = [target]
            for i, candidate in enumerate(candidates_list):
                # TODO: slightly arbitrary choice to use the IQR
                if target[comp_var_target] <= candidates_list[i][comp_var_alt]:
                    top_candidates.append(candidate)
                # is this adding?
                elif target[f'{variable_stem}_mean']-target[f'{variable_stem}_stddev'] <= candidates_list[i][f'{variable_stem}_mean']+candidates_list[i][f'{variable_stem}_stddev']:#candidates_list[i][f'{variable}_p75']:
                    top_candidates.append(candidate)
            top_lags = pd.DataFrame(top_candidates)
        else:
            top_lags = candidates.copy()

        if len(top_lags) > 0:
            top_lags['category'] = category

        return top_lags

    def find_viable_peaks(self, surr_var='neither', y_col='maxlibsize_rho', lag_filter=None, smoothing_window=0):
        """Classify ``self.lag_choices`` into unrestricted/set/anti-set viable-lag categories.

        Step used by :meth:`set_target_lag` (see :meth:`calc_metrics` for
        the full pipeline). Always computes the unrestricted set (all of
        ``self.lag_choices`` via :meth:`lag_is_equivalent`). If
        ``lag_filter`` is given, additionally splits ``self.lag_choices`` by
        whether ``lag_filter(peak_end)`` is true ("set") or false
        ("anti_set") and computes each separately; if not given, the
        unrestricted set is duplicated under the ``'set'`` category instead.

        Parameters
        ----------
        surr_var : str, optional
            Currently unused by this method's body.
        y_col : str, optional
            Metric column stem passed to :meth:`lag_is_equivalent`. Default
            is ``'maxlibsize_rho'``.
        lag_filter : callable, optional
            Predicate over ``peak_end`` values splitting set/anti-set.
            Default is ``None`` (no split).
        smoothing_window : int, optional
            Passed to :meth:`lag_is_equivalent`. Default is ``0``.

        Returns
        -------
        pandas.DataFrame
            Concatenated unrestricted/set/anti-set rows. Also stored as
            ``self.viable_lags``.
        """
        unrestricted_top_lags = self.lag_is_equivalent(self.lag_choices, variable=y_col, category='unrestricted', smoothing_window=smoothing_window)

        if lag_filter is None:
            top_lags = unrestricted_top_lags.copy()
            top_lags['category'] = 'set'
            top_lags = pd.concat([unrestricted_top_lags, top_lags], ignore_index=True)

        else:
            set_candidates = self.lag_choices[self.lag_choices['peak_end'].apply(lag_filter)].copy()
            set_top_lags = self.lag_is_equivalent(set_candidates,  variable=y_col,category='set', smoothing_window=smoothing_window)

            antiset_candidates = self.lag_choices[~self.lag_choices['peak_end'].apply(lag_filter)].copy()
            antiset_top_lags = self.lag_is_equivalent(antiset_candidates,  variable=y_col,category='anti_set', smoothing_window = smoothing_window)

            top_lags = pd.concat([unrestricted_top_lags, set_top_lags, antiset_top_lags], ignore_index=True)

        self.viable_lags = top_lags
        # print('viable lags before surrogate testing:')
        # print(top_lags.head())

        return top_lags


    def calc_lags_peaks(self, relationship_id='r1', surr_var='neither', y_col='maxlibsize_rho', smoothing_window=1):
        """Compute candidate lag peaks for one relationship side from ``self.delta_rho_full``.

        Step used by :meth:`set_target_lag` (see :meth:`calc_metrics` for
        the full pipeline). Loads ``self.delta_rho_full``, summarizes its
        real-data rows via :meth:`_draw_metric_df`, filters to this
        relationship/``surr_var`` combination, and finds candidate peaks via
        :meth:`find_candidate_peaks`.

        Parameters
        ----------
        relationship_id : {'r1', 'r2'}, optional
            Which side's calc-convention name to filter on. Default is ``'r1'``.
        surr_var : str, optional
            Which ``surr_var`` value to filter the real-data rows to.
            Default is ``'neither'`` (non-surrogate rows).
        y_col : str, optional
            Metric column stem passed to :meth:`find_candidate_peaks`.
            Default is ``'maxlibsize_rho'``.
        smoothing_window : int, optional
            Passed to :meth:`find_candidate_peaks`. Default is ``1``.

        Populates
        ---------
        self.lag_choices : pandas.DataFrame
            Candidate peaks from :meth:`find_candidate_peaks`.
        self.real_r_df_tmp : pandas.DataFrame
            The filtered real-data summary used to find them.
        """
        relationship = self.get_relationship(relationship_id=relationship_id).r_calc
        # print(f'calculating candidate peaks for relationship {relationship} with surrogate variable {surr_var} and metric {y_col} (smoothing window={smoothing_window})')
        self.delta_rho_full.get_table()
        gb_real_df = self._draw_metric_df(self.delta_rho_full.real, 'real')
        # print('self.calc_lags_peaks: real performance data frame for peak finding:')
        # print(gb_real_df.head())
        self.delta_rho_full.clear_table()
        real_r_df = gb_real_df[
            (gb_real_df['relation'] == relationship) & (gb_real_df['surr_var'] == surr_var)].reset_index(drop=True)
        print(f'self.calc_lags_peaks: filtered real performance data frame for relationship {relationship} and surrogate variable {surr_var}:')
        print(real_r_df.head())
        all_candidates = self.find_candidate_peaks(real_r_df, y_col=y_col, smoothing_window=smoothing_window)

        self.lag_choices = all_candidates
        self.real_r_df_tmp = real_r_df


    def test_lags(self, lag_df=None, relationship=None, y_col='maxlibsize_rho'):
        """Score each candidate lag's performance against surrogate-data performance.

        Step used by :meth:`set_target_lag` (see :meth:`calc_metrics` for
        the full pipeline). Loads ``self.delta_rho_stats``; if it has no
        surrogate rows, returns ``lag_df`` unchanged (with a count of
        ``0`` rather than testing). Otherwise, for each participating
        variable in turn: counts how many
        surrogate runs of that variable outperform each candidate lag's
        ``{y_col}_mean``, recording ``surr_outperformer_count``/
        ``surr_outperformer_frac``/``surr_count`` per lag and surrogate
        variable.

        Parameters
        ----------
        lag_df : pandas.DataFrame, optional
            Candidate lags to test. Defaults to ``self.viable_lags``.
        relationship : str, optional
            Calc-convention relationship name to filter surrogate
            performance rows to (e.g. from
            ``self.get_relationship(...).r_calc``).
        y_col : str, optional
            Metric column stem. Default is ``'maxlibsize_rho'``.

        Returns
        -------
        pandas.DataFrame
            ``lag_df`` duplicated once per surrogate variable
            (the CedarKit pair or GraphCCM directed pair), each copy's rows annotated with that
            variable's outperformance counts/fractions. Also stored as
            ``self.viable_lags``.
        """
        if lag_df is None:
            lag_df = self.viable_lags.copy()
        # print('viable lag df before surrogate testing:')
        # print(lag_df.head())

        lag_df[['surr_var', 'surr_outperformer_count', 'surr_outperformer_frac', 'surr_count']]=None
        #
        # try:
        #     if self.surr_r_df_tmp is None:
        #         gb_surr_df = self._draw_metric_df('delta_rho_stats', 'surrogate')
        #
        #         # gb_surr = self.delta_rho_stats.surrogate.group_by(["relation", 'lag', 'surr_var', 'surr_num']).aggregate(
        #         #     [("maxlibsize_rho", "mean")])
        #         gb_surr_df = gb_surr.to_pandas()
        #
        # except Exception as e:
        self.delta_rho_stats.get_table()
        print('testing lags against surrogate performance...', self.delta_rho_stats.surrogate.collect().height )
        # if len(self.delta_rho_stats.surrogate) == 0:
        if self.delta_rho_stats.surrogate.collect().height == 0:
            print('No surrogate data found in delta_rho_stats.surrogate for testing lags.')
            self.viable_lags =lag_df
            return lag_df
        gb_surr = self.delta_rho_stats.surrogate.group_by(
            ["relation", "lag", "surr_var", "surr_num"]
        ).agg(
            pl.col("maxlibsize_rho").mean().alias("maxlibsize_rho_mean")
        )
        gb_surr_df = gb_surr.collect().to_pandas() if isinstance(gb_surr, pl.LazyFrame) else gb_surr.to_pandas()
        self.delta_rho_stats.clear_table()
        # print(gb_surr_df.head())

        # self.delta_rho_stats.get_table()
        # gb_surr = self.delta_rho_stats.surrogate.group_by(["relation", 'lag', 'surr_var', 'surr_num']).aggregate([("maxlibsize_rho", "mean")])
        # gb_surr_df = gb_surr.to_pandas()
        # self.delta_rho_stats.clear_table()
        # print('surrogate performance data frame for testing:')
        # print(gb_surr_df.head())

        if isinstance(self.relationships, Relationship):
            surrogate_variables = [self.relationships.var_x, self.relationships.var_y]
        else:
            # GraphCM persists the configured generic surrogate prefix (for
            # example ``d18O``), not its variable ID. Those are the values in
            # the output table and therefore the only valid filter keys here.
            surrogate_variables = sorted(
                value for value in gb_surr_df["surr_var"].dropna().unique()
                if value != "neither"
            )
            if not surrogate_variables:
                raise NotImplementedError(
                    "GraphCM lag metrics require at least one surrogate variable in the output."
                )

        lag_performance_surr_tests = []
        for surr_var in surrogate_variables:
            lag_df__surr_test = lag_df.copy()
            lag_df__surr_test['surr_var'] = surr_var
            surr_rx_df = gb_surr_df[(gb_surr_df['relation'] == relationship) & (gb_surr_df['surr_var'] == surr_var)]
            # print(f'surrogate performance for variable {surr_var}:')
            # print(surr_rx_df.head())
            surr_rx_count = len(surr_rx_df.surr_num.unique())
            for ik in range(len(lag_df__surr_test)):
                surr_rx_df_outperformers = surr_rx_df[surr_rx_df[f'{y_col}_mean'] > lag_df__surr_test.iloc[ik][f'{y_col}_mean']]
                # print(f'lag {lag_df__surr_test.iloc[ik]["lag"]} has {len(surr_rx_df_outperformers)} outperforming surrogates for variable {surr_var}')
                surr_rx_df_outperformers_count = len(surr_rx_df_outperformers.surr_num.unique())
                lag_df__surr_test.at[ik, 'surr_outperformer_count'] = surr_rx_df_outperformers_count
                lag_df__surr_test.at[ik, 'surr_outperformer_frac'] = surr_rx_df_outperformers_count / surr_rx_count if surr_rx_count > 0 else None
                lag_df__surr_test.at[ik, 'surr_count'] = surr_rx_count
                # print(lag_df__surr_test.loc[ik,:])
            lag_performance_surr_tests.append(lag_df__surr_test)


        lag_performance_surr_tested_df = pd.concat(lag_performance_surr_tests, ignore_index=True)
        # print('lag end', lag_performance_surr_tested_df)
        self.viable_lags = lag_performance_surr_tested_df
        return lag_performance_surr_tested_df

    def set_target_lag(self, relationship_id = 'r1', y_col='maxlibsize_rho', smoothing_window=1, lag=None):
        """Choose the single best lag for one relationship side, after candidate-finding and surrogate testing.

        Step used by :meth:`_calc_metrics` (see :meth:`calc_metrics` for the
        full pipeline). Ensures ``self.lag_choices`` exists (computing it via
        :meth:`calc_lags_peaks` if not), then runs
        :meth:`find_viable_peaks` → :meth:`test_lags` to get scored
        candidates, restricts to the ``'unrestricted'`` category, and picks
        the lag with the highest ``maxlibsize_rho_mean`` among those sorted
        by absolute lag (closest-to-zero first) and decision metric — by
        convention, a positive lag is assumed to put the target variable
        ahead of the predictor (see the inline comment on this assumption).

        Parameters
        ----------
        relationship_id : {'r1', 'r2'}, optional
            Which side to pick a target lag for. Default is ``'r1'``.
        y_col : str, optional
            Metric column stem. Default is ``'maxlibsize_rho'``.
        smoothing_window : int, optional
            Passed through to :meth:`find_viable_peaks`. Default is ``1``.
        lag : None or callable or {'pos', 'neg'}, optional
            Passed to :meth:`_resolve_metrics_lag_filter` (note: the
            resulting filter is not actually applied to
            :meth:`find_viable_peaks` in this method's body — see notes doc).

        Returns
        -------
        int or float or None
            The chosen lag, or ``None`` if no unrestricted viable lags were
            found (also sets ``self.target_lag = None`` in that case).
            Also sets ``self.viable_lags``.
        """
        relationship = self.get_relationship(relationship_id=relationship_id).r_calc
        lag_filter = self._resolve_metrics_lag_filter(lag=lag)
        # print(f'setting target lag for relationship {relationship} using metric {y_col} with lag filter {lag} and smoothing window {smoothing_window}')
        # print(f'setting target lag for relationship {relationship} using metric {y_col} with lag filter {lag} and smoothing window {smoothing_window}')

        if hasattr(self, 'lag_choices') is False:
            # print('lag_choices attribute not found, initializing to None')
            self.lag_choices = None

        if self.lag_choices is None:
            print('lag choices not found, calculating candidate peaks...')
            self.calc_lags_peaks(relationship_id=relationship_id, y_col=y_col, smoothing_window=smoothing_window)

        viable_lags  = self.find_viable_peaks(lag_filter=None, y_col=y_col, smoothing_window=smoothing_window)
        # print('viable lags after candidate peak finding:')
        # print(viable_lags.head())
        tested_viable_lags = self.test_lags(lag_df = viable_lags, relationship=relationship)
        print('viable lags after surrogate testing:')
        print(tested_viable_lags.head())

        decision_metric = f'{y_col}_mean'
        if smoothing_window > 1:
            decision_metric = f'{decision_metric}_smooth'

        tested_viable_lags['abs_lag'] = tested_viable_lags['lag'].apply(lambda x: abs(x) if pd.notna(x) else np.inf)

        unrestricted_lags = tested_viable_lags[tested_viable_lags['category'] == 'unrestricted'].copy()
        print('unrestricted lags after surrogate testing:')
        print(unrestricted_lags.head())
        if unrestricted_lags.empty:
            self.viable_lags = unrestricted_lags
            self.target_lag = None
            print(f'No unrestricted lags available for relationship {relationship}; target lag remains None')
            return None
        unrestricted_lags_filtered = unrestricted_lags.copy().drop(columns=['surr_var']).drop_duplicates(subset=['lag']).sort_values(by=[ 'abs_lag', 'lag',decision_metric ],
                                                          ascending=[True, False, False])

        sorted_unrestricted_lags = pd.concat([unrestricted_lags_filtered[unrestricted_lags_filtered['lag'] >= 0].copy(),
                                       unrestricted_lags_filtered[unrestricted_lags_filtered['lag'] < 0].copy()])
        if sorted_unrestricted_lags.empty:
            self.viable_lags = unrestricted_lags
            self.target_lag = None
            print(f'No sorted unrestricted lags available for relationship {relationship}; target lag remains None')
            return None
        # print('sorted unrestricted lags:')
        # print(sorted_unrestricted_lags.head())
        # this is convention; assumes positive lag puts target ahead of col... if shift is set up for col behind target, this is wrong
        target_lag = sorted_unrestricted_lags.sort_values(by=['maxlibsize_rho_mean'], ascending=False)['lag'].values[0]
        # if len(sorted_unrestricted_lags[sorted_unrestricted_lags['peak_end']>=0])>0:
        #     target_lag = sorted_unrestricted_lags[sorted_unrestricted_lags['peak_end']>=0].iloc[0]['lag']
        #     print('target lag after surrogate testing:')
        #     print(target_lag)
        # else:
        #     target_lag = sorted_unrestricted_lags.iloc[0]['lag']

        self.viable_lags = unrestricted_lags
        self.target_lag = target_lag
        print('self.target_lag', self.target_lag)

        return target_lag


    def _calc_metrics(self, relationship_id='r1', lag=None, smoothing_window=1, y_col='maxlibsize_rho'):
        """Set one relationship side's scalar summary attributes from its target lag's data row.

        Called by :meth:`calc_metrics` (see that method's docstring for the
        full pipeline). Ensures ``self.target_lag`` is set (via
        :meth:`set_target_lag` if not — returning ``None`` early if no
        target lag or no matching row in ``self.viable_lags`` is found),
        then writes ``delta_rho``/``maxlibsize_rho``/``lag``/surrogate
        outperformance counts and fractions/``peak_start``/``peak_end``
        directly onto ``self.r1`` or ``self.r2`` (accessed directly, not via
        :meth:`get_relationship` — see the caveat about whether ``r1``/
        ``r2`` are guaranteed populated, noted in ``__init__``'s docstring).

        Parameters
        ----------
        relationship_id : {'r1', 'r2'}, optional
            Which side to compute and assign metrics for. Default is ``'r1'``.
        lag : None or callable or {'pos', 'neg'}, optional
            Passed through to :meth:`set_target_lag` if a target lag still
            needs to be found.
        smoothing_window : int, optional
            Passed through to :meth:`set_target_lag`. Default is ``1``.
        y_col : str, optional
            Passed through to :meth:`set_target_lag`. Default is ``'maxlibsize_rho'``.

        Returns
        -------
        None
            Always returns ``None`` — this method communicates exclusively
            via the side effect of setting attributes on ``self.r1``/``self.r2``.
        """
        relationship = self.get_relationship(relationship_id=relationship_id).r_calc

        if hasattr(self, "target_lag") is False:
            self.target_lag = None

        if self.target_lag is None:
            target_lag = self.set_target_lag(relationship_id=relationship_id, y_col=y_col, smoothing_window=smoothing_window, lag=lag)
            if target_lag is None:
                print(f'No target lag available for relationship {relationship}; skipping metric assignment')
                return None

        target_lag_info = self.viable_lags[self.viable_lags['lag'] == self.target_lag].copy()
        print('target_lag_info', target_lag_info)
        if target_lag_info.empty:
            print(f'No target lag info rows available for relationship {relationship}; skipping metric assignment')
            return None

        target_lag_row = target_lag_info.iloc[0]
        if isinstance(self.relationships, Relationship):
            surrogate_variables = [self.relationships.var_x, self.relationships.var_y]
        else:
            surrogate_variables = list(getattr(self.relationships, "participant_variables", ()))
            if len(surrogate_variables) != 2:
                raise NotImplementedError(
                    "CedarKit lag metrics currently require exactly two directed participants."
                )

        target_lag_by_surr = (
            target_lag_info
            .drop_duplicates(subset='surr_var')
            .set_index('surr_var')
            .reindex(surrogate_variables)
        )

        surr_rx_count = target_lag_by_surr.loc[surrogate_variables[0], 'surr_count']
        surr_rx_df_outperformers_count = target_lag_by_surr.loc[surrogate_variables[0], 'surr_outperformer_count']
        surr_ry_count = target_lag_by_surr.loc[surrogate_variables[1], 'surr_count']
        surr_ry_df_outperformers_count = target_lag_by_surr.loc[surrogate_variables[1], 'surr_outperformer_count']

        target_relationship = self.r1 if relationship_id == 'r1' else self.r2

        target_relationship.surr_rx_count = surr_rx_count
        target_relationship.surr_rx_count_outperforming = surr_rx_df_outperformers_count
        target_relationship.surr_ry_count = surr_ry_count
        target_relationship.surr_ry_count_outperforming = surr_ry_df_outperformers_count
        target_relationship.delta_rho = target_lag_row['delta_rho_mean']
        target_relationship.maxlibsize_rho = target_lag_row['maxlibsize_rho_mean']
        target_relationship.lag = target_lag_row['lag']
        target_relationship.surr_rx_outperforming_frac = (
            surr_rx_df_outperformers_count / surr_rx_count
            if pd.notna(surr_rx_count) and surr_rx_count > 0
            else None
        )
        target_relationship.surr_ry_outperforming_frac = (
            surr_ry_df_outperformers_count / surr_ry_count
            if pd.notna(surr_ry_count) and surr_ry_count > 0
            else None
        )
        target_relationship.peak_start = target_lag_row['peak_start']
        target_relationship.peak_end = target_lag_row['peak_end']

        # target_lag_info = self.viable_lags[self.viable_lags['lag'] == self.target_lag].copy()
        # print('target_lag_info', target_lag_info)
        #
        # surr_rx_count = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_x]['surr_count'].iloc[0]
        # surr_rx_df_outperformers_count = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_x]['surr_outperformer_count'].iloc[0]
        # surr_ry_count = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_y]['surr_count'].iloc[0]
        # surr_ry_df_outperformers_count = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_y]['surr_outperformer_count'].iloc[0]
        #
        # delta_rho_mean = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_x]['delta_rho_mean'].iloc[0]
        # maxlibsize_rho_mean = target_lag_info[target_lag_info['surr_var'] == self.relationships.var_x]['maxlibsize_rho_mean'].iloc[0]
        # lag_value = target_lag_info['lag'].iloc[0]
        # peak_start = target_lag_info['peak_start'].iloc[0]
        # peak_end = target_lag_info['peak_end'].iloc[0]
        #
        # if relationship_id == 'r1':
        #     self.r1.surr_rx_count = surr_rx_count
        #     self.r1.surr_rx_count_outperforming = surr_rx_df_outperformers_count
        #     self.r1.surr_ry_count = surr_ry_count
        #     self.r1.surr_ry_count_outperforming = surr_ry_df_outperformers_count
        #     self.r1.delta_rho = delta_rho_mean
        #     self.r1.maxlibsize_rho = maxlibsize_rho_mean
        #     self.r1.lag = lag_value
        #     self.r1.surr_rx_outperforming_frac = surr_rx_df_outperformers_count / surr_rx_count if surr_rx_count > 0 else None
        #     self.r1.surr_ry_outperforming_frac = surr_ry_df_outperformers_count / surr_ry_count if surr_ry_count > 0 else None
        #     self.r1.peak_start = peak_start
        #     self.r1.peak_end = peak_end
        # elif relationship_id == 'r2':
        #     self.r2.surr_rx_count = surr_rx_count
        #     self.r2.surr_rx_count_outperforming = surr_rx_df_outperformers_count
        #     self.r2.surr_ry_count = surr_ry_count
        #     self.r2.surr_ry_count_outperforming = surr_ry_df_outperformers_count
        #     self.r2.delta_rho = delta_rho_mean
        #     self.r2.maxlibsize_rho = maxlibsize_rho_mean
        #     self.r2.lag = lag_value
        #     self.r2.surr_rx_outperforming_frac = surr_rx_df_outperformers_count / surr_rx_count if surr_rx_count > 0 else None
        #     self.r2.surr_ry_outperforming_frac = surr_ry_df_outperformers_count / surr_ry_count if surr_ry_count > 0 else None
        #     self.r2.peak_start = peak_start
        #     self.r2.peak_end = peak_end


    def calc_delta_rho(self, *, stats_out=True, full_out=False, **kwargs):
        """Compute delta-rho stats/full outputs per calc-group, from ``self.table``.

        Loads ``self.table.full``, determines which trait columns vary
        *within* a `LibSize` group (via ``self.grp_config.trait_hierarchy``)
        versus which define the group itself (``calc_grp_cols`` — the
        complement, plus ``'relation'`` if present), then for each unique
        combination of ``calc_grp_cols`` values, filters to that group's
        sub-table and applies :func:`compute_delta_rho_grp`. Results across
        groups are concatenated and stored.

        Parameters
        ----------
        stats_out : bool, optional
            Whether to compute and store ``self.delta_rho_stats``. Default
            is ``True``.
        full_out : bool, optional
            Whether to compute and store ``self.delta_rho_full``. Default
            is ``False``.
        **kwargs
            Passed through to :func:`compute_delta_rho_grp` (e.g.
            ``best_window_halfwidth``, ``min_window``, ``max_window``).

        Returns
        -------
        OutputCollection
            ``self``, for chaining.

        Raises
        ------
        TypeError
            If ``self.table.full`` isn't one of the supported table types.
        """
        full = self.table.full
        if isinstance(full, pl.LazyFrame):
            full = full.collect()
        elif isinstance(full, pd.DataFrame):
            full = pl.from_pandas(full)
        elif not isinstance(full, pl.DataFrame):
            raise TypeError(f"Unsupported full table type: {type(full)}")

        if isinstance(self.grp_config, RunConfig):
            group_traits_below = self.grp_config.trait_hierarchy(
                full, 'LibSize', level="below", threshold=0.8, include_ids=True
            )
            calc_grp_cols = [
                col for col in full.columns
                if col in self.grp_config.traits and col not in group_traits_below
            ]
        else:
            graph_traits = set(getattr(self.grp_config, 'traits', ()))
            calc_grp_cols = [
                col for col in full.columns
                if col in graph_traits
                and col not in {'run_config_id', 'run_set_id', 'draw_id', 'draw_size'}
                and full.get_column(col).null_count() < full.height
            ]
            for col in ('relation', 'metric', 'metric_mask_id'):
                if col in full.columns and col not in calc_grp_cols:
                    calc_grp_cols.append(col)

        if 'relation' in full.columns:
            if 'relation' not in calc_grp_cols:
                calc_grp_cols.append('relation')

        unique_tbl = full.select(calc_grp_cols).unique(maintain_order=True)

        stats_tables = []
        full_tables = []
        for row in unique_tbl.iter_rows(named=True):
            gd = {col: correct_iterable(row[col]) for col in calc_grp_cols}
            filters = [pl.col(col).is_in(correct_iterable(row[col])) for col in calc_grp_cols]
            grp_tbl = full.filter(reduce(operator.and_, filters))

            s_tbl, f_tbl = compute_delta_rho_grp(
                grp_tbl, gd, stats=stats_out, full=full_out, **kwargs
            )
            if stats_out is True and s_tbl is not None:
                stats_tables.append(s_tbl)
            if full_out is True and f_tbl is not None:
                full_tables.append(f_tbl)

        if stats_out is True:
            out_stats = pl.concat(stats_tables, how='diagonal_relaxed') if stats_tables else None
            self.delta_rho_stats = Output(out_stats, outtype='delta_rho_stats', tmp_dir=self.tmp_path)#, use_case='delta_rho_stats')
        if full_out is True:
            out_full = pl.concat(full_tables, how='diagonal_relaxed') if full_tables else None
            self.delta_rho_full = Output(out_full, outtype='delta_rho_full', tmp_dir=self.tmp_path)#, use_case='delta_rho_full')

        return self

    def aggregate_libsize(self, query_config=None): #process_group_table
        """Average numeric columns across draws above the ``knn`` threshold.

        Resolves ``knn`` from ``query_config`` (if given) or
        ``self.grp_config``; no-ops (returns ``self`` unchanged) if `knn`
        can't be resolved, if `self.table` has no ``LibSize`` column, or if
        no rows have ``LibSize > knn + 1``. CedarKit ``RunConfig`` retains
        its legacy literal grouping. A GraphCM configuration derives grouping
        columns from its declared traits, while excluding exact run/draw IDs
        so numeric values are averaged across draws. Both paths retain
        ``LibSize`` and ``relation`` when present.

        Parameters
        ----------
        query_config : RunConfig, optional
            If given, its ``knn`` is used instead of ``self.grp_config.knn``.

        Returns
        -------
        OutputCollection
            ``self``, for chaining. Sets ``self.libsize_aggregated`` as a
            side effect when aggregation actually runs.

        Raises
        ------
        TypeError
            If ``self.table.full`` isn't one of the supported table types.
        """
        knn = get_static(query_config.knn if query_config is not None else self.grp_config.knn)
        if knn is None:
            return self
        try:
            knn = int(knn)
        except (TypeError, ValueError):
            knn = int(float(knn))

        full = self.table.full
        if isinstance(full, pl.LazyFrame):
            full = full.collect()
        elif isinstance(full, pd.DataFrame):
            full = pl.from_pandas(full)
        elif not isinstance(full, pl.DataFrame):
            raise TypeError(f"Unsupported full table type: {type(full)}")

        if "LibSize" not in full.columns:
            return self

        full = full.with_columns(pl.col("LibSize").cast(pl.Float64, strict=False))
        group_table = full.filter(pl.col("LibSize") > (knn + 1))
        if group_table.height == 0:
            return self

        if isinstance(self.grp_config, RunConfig):
            calc_grp_cols = [
                'E', 'tau', 'Tp', 'lag', 'knn', 'surr_var', 'surr_num',
                'x_id', 'x_age_model_ind', 'x_var', 'y_id',
                'y_age_model_ind', 'y_var', 'LibSize', 'ind_i',
                'relation', 'forcing', 'responding',
            ]
        else:
            graph_traits = set(getattr(self.grp_config, 'traits', ()))
            calc_grp_cols = [
                col for col in full.columns
                if col in graph_traits and col not in {'run_config_id', 'run_set_id', 'draw_id'}
                and full.get_column(col).null_count() < full.height
            ]
            for col in ('relation', 'metric', 'metric_mask_id'):
                if col in full.columns and col not in calc_grp_cols:
                    calc_grp_cols.append(col)

        if "LibSize" in full.columns and "LibSize" not in calc_grp_cols:
            calc_grp_cols.append("LibSize")
        if 'relation' in full.columns and 'relation' not in calc_grp_cols:
            calc_grp_cols.append('relation')
        calc_grp_cols = [col for col in calc_grp_cols if col in full.columns]

        aggregated_cols = [
            col for col, dtype in full.schema.items()
            if (col not in calc_grp_cols)
            and ('id' not in col)
            and ('ind' not in col)
            and dtype.is_numeric()
        ]
        log_line(self.log, ['aggregated cols', aggregated_cols],
                 indent=0, log_type="debug")

        if len(aggregated_cols) == 0:
            grouped_aggregated_table = group_table.select(calc_grp_cols).unique(maintain_order=True)
        else:
            grouped_aggregated_table = group_table.group_by(calc_grp_cols).agg(
                [pl.col(col).mean().alias(col) for col in aggregated_cols]
            )

        self.libsize_aggregated = Output(grouped_aggregated_table, outtype='libsize_aggregated', tmp_dir=self.tmp_path)#, use_case='libsize_aggregated')
        return self

    def clear_tables(self):
        """Release memory held by every populated ``Output`` attribute and Arrow pools.

        Calls ``clear_table()`` on each of ``table``, ``libsize_aggregated``,
        ``active_stats``, ``active_full``, ``delta_rho_stats``,
        ``delta_rho_full`` that's currently set, then runs garbage
        collection.
        """
        if hasattr(self, "table") and self.table is not None:
            self.table.clear_table()

        if hasattr(self, "libsize_aggregated") and self.libsize_aggregated is not None:
            self.libsize_aggregated.clear_table()

        if hasattr(self, "active_stats") and self.active_stats is not None:
            self.active_stats.clear_table()

        if hasattr(self, "active_full") and self.active_full is not None:
            self.active_full.clear_table()

        if hasattr(self, "delta_rho_stats") and self.delta_rho_stats is not None:
            self.delta_rho_stats.clear_table()

        if hasattr(self, "delta_rho_full") and self.delta_rho_full is not None:
            self.delta_rho_full.clear_table()

        gc.collect()

    def get_table_paths(self):
        """Collect the file path of each populated ``Output`` attribute that has one.

        Returns
        -------
        dict[str, pathlib.Path]
            Maps attribute name (``'table'``, ``'libsize_aggregated'``,
            ``'delta_rho_stats'``, ``'delta_rho_full'`` — notably not
            ``'active_stats'``/``'active_full'``, which this method doesn't
            check) to that ``Output``'s ``path``, for whichever of those are
            set and have a non-``None`` path.
        """
        paths = {}
        if hasattr(self, "table") and self.table is not None and self.table.path is not None:
            paths['table'] = self.table.path
        if hasattr(self, "libsize_aggregated") and self.libsize_aggregated is not None and self.libsize_aggregated.path is not None:
            paths['libsize_aggregated'] = self.libsize_aggregated.path
        if hasattr(self, "delta_rho_stats") and self.delta_rho_stats is not None and self.delta_rho_stats.path is not None:
            paths['delta_rho_stats'] = self.delta_rho_stats.path
        if hasattr(self, "delta_rho_full") and self.delta_rho_full is not None and self.delta_rho_full.path is not None:
            paths['delta_rho_full'] = self.delta_rho_full.path
        return paths

    def resolve_paths(self, dyad_dir, *, prefer_local=True):
        """Rebase paths for this collection beneath ``dyad_dir``.

        This is the explicit counterpart to :meth:`Output.resolve_path` for
        callers (such as the object-grid runner) that already know the local
        dyad directory.  It preserves every nested path component below the
        dyad rather than assuming that all files live directly in ``tmp``.
        Local counterparts are preferred by default because an explicit
        ``dyad_dir`` identifies the active local analysis.  Pass
        ``prefer_local=False`` to retain a still-reachable stored path.

        Returns
        -------
        int
            Number of paths successfully rebound.
        """
        dyad_dir = Path(dyad_dir)
        rebound = 0
        self.dyad_home = dyad_dir.parent

        for attr in ['table', 'libsize_aggregated', 'active_stats', 'active_full',
                     'delta_rho_stats', 'delta_rho_full']:
            output = getattr(self, attr, None)
            if output is None or not hasattr(output, 'resolve_path'):
                continue
            old_path = output.path
            new_path = output.resolve_path(dyad_dir, prefer_local=prefer_local)
            if new_path != old_path:
                rebound += 1

        if self.grp_config is not None and getattr(self.grp_config, 'output_path', None):
            paths = self.grp_config.output_path
            for index, old_path in enumerate(paths):
                new_path = resolve_dyad_anchored_path(old_path, dyad_dir, prefer_local=prefer_local)
                if new_path != old_path:
                    paths[index] = new_path
                    rebound += 1
        return rebound

    def migrate_path(self, new_dyad_home=None, tmp_home=None):
        """Re-point every populated ``Output`` attribute's path/``tmp_dir`` at a new dyad home.

        Because ``Output`` tables are read in and cleared rather than held
        open continuously (see :meth:`Output.clear_table`), their file
        paths need to be explicitly updated whenever the dyad home or
        temporary directory changes — this method does that for every
        populated ``Output`` attribute (``table``, ``libsize_aggregated``,
        ``active_stats``, ``active_full``, ``delta_rho_stats``,
        ``delta_rho_full``), preserving each file's name but rehoming it
        under the new ``self.tmp_path``.

        Parameters
        ----------
        new_dyad_home : str or pathlib.Path, optional
            Path to the parent of this variable pair's ("dyad's") directory
            (e.g. the directory housing ``Erb22daGMST_Wu18TSI``). Defaults
            to ``self.dyad_home``, or ``self.tmp_path.parent.parent`` if
            that's also unset.
        tmp_home : str, optional
            Name of the dyad directory itself (e.g. ``Erb22daGMST_Wu18TSI``).
            Defaults to ``self.tmp_path.parent.name``.
        """
        if new_dyad_home is None:
            new_dyad_home = self.dyad_home
        if new_dyad_home is None:
            new_dyad_home = self.tmp_path.parent.parent

        self.dyad_home = new_dyad_home

        if tmp_home is None:
            tmp_home = self.tmp_path.parent.name

        self.tmp_path = self.dyad_home / tmp_home / 'tmp'

        if hasattr(self, "table") and self.table is not None and self.table.path is not None:
            self.table.path = self.tmp_path / self.table.path.name
            self.table.tmp_dir = self.tmp_path
        if hasattr(self, "libsize_aggregated") and self.libsize_aggregated is not None and self.libsize_aggregated.path is not None:
            self.libsize_aggregated.path = self.tmp_path / self.libsize_aggregated.path.name
            self.libsize_aggregated.tmp_dir = self.tmp_path
        if hasattr(self, "active_stats") and self.active_stats is not None and self.active_stats.path is not None:
            self.active_stats.path = self.tmp_path / self.active_stats.path.name
            self.active_stats.tmp_dir = self.tmp_path
        if hasattr(self, "active_full") and self.active_full is not None and self.active_full.path is not None:
            self.active_full.path = self.tmp_path / self.active_full.path.name
            self.active_full.tmp_dir = self.tmp_path
        if hasattr(self, "delta_rho_stats") and self.delta_rho_stats is not None and self.delta_rho_stats.path is not None:
            self.delta_rho_stats.path = self.tmp_path / self.delta_rho_stats.path.name
            self.delta_rho_stats.tmp_dir = self.tmp_path
        if hasattr(self, "delta_rho_full") and self.delta_rho_full is not None and self.delta_rho_full.path is not None:
            self.delta_rho_full.path = self.tmp_path / self.delta_rho_full.path.name
            self.delta_rho_full.tmp_dir = self.tmp_path


def merge_variable_ts(col_var_obj, target_var_obj):
    """Inner-join two ``VarObject`` timeseries on their shared time column.

    Renames each object's value column from its raw ``col_name`` to its
    variable name (``var``), then merges on ``col_var_obj.time_var`` (an
    inner join, so only overlapping timestamps survive). If the merge fails
    due to a time-column dtype mismatch, both time columns are coerced to a
    common type first — ``float`` if either object's ``delta_ts`` is a
    numeric type, otherwise ``int`` — and the merge is retried.

    Parameters
    ----------
    col_var_obj : VarObject
        Predictor variable, with ``.ts``, ``.col_name``, ``.var``, and
        ``.time_var`` already populated (e.g. via ``pull_ts``).
    target_var_obj : VarObject
        Target variable, with the same attributes populated.

    Returns
    -------
    pandas.DataFrame
        Merged frame with columns ``[time_var, col_var_obj.var,
        target_var_obj.var]``, sorted by the time column.
    """
    col_df = col_var_obj.ts.rename(columns={col_var_obj.col_name: col_var_obj.var})
    target_df = target_var_obj.ts.rename(columns={target_var_obj.col_name: target_var_obj.var})
    try:
        merged_df = pd.merge(col_df, target_df, on=col_var_obj.time_var, how='inner')
    except:
        time_types = [type(col_var_obj.delta_ts), type(target_var_obj.delta_ts)]
        if any([t in [int, float, np.int64, np.float64] for t in time_types]):
            col_df[col_var_obj.time_var] = col_df[col_var_obj.time_var].astype(float)
            target_df[target_var_obj.time_var] = target_df[target_var_obj.time_var].astype(float)
        else:
            col_df[col_var_obj.time_var] = col_df[col_var_obj.time_var].astype(int)
            target_df[target_var_obj.time_var] = target_df[target_var_obj.time_var].astype(int)
        merged_df = pd.merge(col_df, target_df, on=col_var_obj.time_var, how='inner')

    df = merged_df.sort_values(by=col_var_obj.time_var).reset_index(drop=True)
    return df


class CMConfigBase(RunConfig):
    """Shared base for cross-mapping configs: resolves variable objects and builds a merged dataframe.

    Extends :class:`RunConfig` with the machinery common to running any
    cross-mapping analysis between two resolved variables: pulling each
    variable's real or surrogate timeseries (:meth:`set_col_ts`/
    :meth:`set_target_ts`) and merging them into one working dataframe
    (:meth:`make_df`, via :func:`merge_variable_ts`). :class:`CCMConfig` is
    the (currently only) subclass, adding CCM-specific run setup on top.

    This base class holds logic that was recently consolidated out of
    :class:`CCMConfig` so it isn't duplicated if/when other cross-mapping
    config types are added — see the notes doc for ``CCMConfig`` methods
    that still locally override these and are candidates for deletion now
    that the shared versions live here.
    """

    def __init__(self, grp_specs, config, proj_dir=None, tmp_dir=None, exclusion_radius=None, init_var_objs=True):
        """Build the underlying ``RunConfig``, resolve variable objects, and compute ``exclusion_radius``.

        Parameters
        ----------
        grp_specs : dict
            Trait values, passed to ``RunConfig.__init__``.
        config : cedarkit.core.project_config.ProjectConfig
            Project configuration, passed to :meth:`RunConfig.set_var_objs`
            if ``init_var_objs`` is true.
        proj_dir : str or pathlib.Path, optional
            Root project directory. Overrides ``self.proj_dir`` if given
            (otherwise whatever ``RunConfig.__init__``/``populate`` already
            set it to, typically ``None``, is kept).
        tmp_dir : str or pathlib.Path, optional
            Temporary directory, passed to ``RunConfig.__init__``.
        exclusion_radius : float, optional
            CCM exclusion radius. If not given, computed as
            ``abs(tau * (E - 1))`` from the resolved ``self.tau``/``self.E``.
        init_var_objs : bool, optional
            Whether to resolve ``col_var_obj``/``target_var_obj`` via
            :meth:`RunConfig.set_var_objs` (only happens if ``self.proj_dir``
            is also set). Default is ``True``.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        super().__init__(grp_specs, tmp_dir=tmp_dir)

        if proj_dir is not None:
            self.proj_dir = proj_dir

        if init_var_objs and self.proj_dir is not None:
            self.set_var_objs(config, self.proj_dir)

        self.df = None
        self.output_dir = None
        self.output_path = None
        self.calc_location = None
        self.time_var = None
        self.noTime = None

        self.exclusion_radius = np.abs(get_static(self.tau) * (get_static(self.E) - 1)) if exclusion_radius is None else exclusion_radius

    def set_col_ts(self, surr_num=None):
        """Load ``self.col_var_obj``'s real or surrogate timeseries into it, in place.

        No-op if ``self.col_var_obj`` is unset. If it's already configured
        for surrogate data (``ts_type == 'surr'``) and doesn't yet have a
        ``surr_num``, adopts ``surr_num`` for it. Then loads real data if
        ``surr_num`` is ``0`` or unset, otherwise loads that surrogate run.

        Parameters
        ----------
        surr_num : int, optional
            Surrogate run number to adopt if ``self.col_var_obj`` doesn't
            already have one set.
        """
        if getattr(self, "col_var_obj", None) is None:
            return
        if self.col_var_obj.ts_type == 'surr':
            if (self.col_var_obj.surr_num is None) and (surr_num is not None):
                self.col_var_obj.surr_num = surr_num

        if self.col_var_obj.surr_num not in (0, None):
            self.col_var_obj.get_surr(self.col_var_obj.surr_num)
        else:
            self.col_var_obj.get_real()

    def set_target_ts(self, surr_num=None):
        """Load ``self.target_var_obj``'s real or surrogate timeseries into it, in place.

        No-op if ``self.target_var_obj`` is unset. If ``self.surr_var``
        indicates this run uses surrogate target data (matches ``'y'``,
        ``self.target_var``, or ``'both'``) and ``surr_num`` is given,
        adopts it. Then loads real data if the resulting ``surr_num`` is
        ``0`` or unset, otherwise loads that surrogate run.

        Note: ``CCMConfig`` currently overrides this method with logic that
        behaves slightly differently in the case where ``self.surr_var``
        matches but ``surr_num`` is *not* given — see the notes doc before
        assuming the override can be deleted as a pure duplicate.

        Parameters
        ----------
        surr_num : int, optional
            Surrogate run number to adopt if ``self.surr_var`` indicates the
            target side should use surrogate data.
        """
        if getattr(self, "target_var_obj", None) is None:
            return
        if self.surr_var in ('y', self.target_var, 'both'):
            if surr_num is not None:
                self.target_var_obj.surr_num = surr_num

        if self.target_var_obj.surr_num not in (0, None):
            self.target_var_obj.get_surr(self.target_var_obj.surr_num)
        else:
            self.target_var_obj.get_real()

    def make_df(self):
        """Merge ``col_var_obj``/``target_var_obj``'s timeseries into ``self.df``, sliced to the training range.

        Sets ``self.df`` via :func:`merge_variable_ts`, then slices it to
        ``[self.train_ind_i : self.train_ind_f]`` (or just
        ``[self.train_ind_i:]`` if ``self.train_ind_f`` is ``None``).

        Returns
        -------
        CMConfigBase
            ``self``, for chaining (e.g. ``self.make_df().shift()`` in
            :class:`CCMConfig`).
        """
        self.df = merge_variable_ts(self.col_var_obj, self.target_var_obj)
        self.df = self.df.iloc[self.train_ind_i : self.train_ind_f].reset_index(drop=True) if self.train_ind_f is not None else self.df.iloc[self.train_ind_i : ].reset_index(drop=True)
        return self


class CCMConfig(CMConfigBase):
    """Fully resolved configuration for one CCM run, ready to execute via :meth:`run_ccm`.

    Extends :class:`CMConfigBase` with everything specific to actually
    running CCM: the output filename/path (:meth:`get_filename`,
    :meth:`set_output_calc_sub`), the libsize range to sweep
    (:meth:`set_libsizes`), and the lag-shift applied to the merged
    dataframe (:meth:`shift`) before :meth:`run_ccm` hands off to
    ``cedarkit.utils.experiments.run_experiment``.

    ``set_col_ts``/``set_target_ts``/``make_df`` are also defined directly
    on this class, overriding the versions on :class:`CMConfigBase` that
    were recently consolidated there. ``set_col_ts``/``make_df`` are exact
    duplicates of the base class versions and are candidates for deletion;
    ``set_target_ts`` is *not* a pure duplicate — see the notes doc before
    removing it.
    """

    def __init__(self, grp_specs, config, proj_dir=None, cpus=1, exclusion_radius=None, limit_surr_libsizes= True):
        """Resolve variables, build the merged/shifted dataframe, and determine the output path and libsize range.

        Builds on :class:`CMConfigBase`'s variable resolution: computes the
        output filename/path (:meth:`get_filename`,
        :meth:`set_output_calc_sub`), loads both variables' timeseries
        (:meth:`set_col_ts`/:meth:`set_target_ts`), merges and shifts them
        into ``self.df`` (:meth:`make_df` then :meth:`shift`), resolves the
        libsize sweep range (:meth:`set_libsizes`, trimmed to the last 5 if
        either variable is surrogate data and ``limit_surr_libsizes`` is
        true), and infers ``self.time_var``/``self.noTime`` from whichever
        column of ``self.df`` isn't one of the two variable columns.

        Parameters
        ----------
        grp_specs : dict
            Trait values for this run. Also used to build a separate,
            unresolved ``RunConfig`` snapshot stored as ``self.rc`` (used
            later by :meth:`run_ccm` to construct the output
            ``OutputCollection``).
        config : cedarkit.core.project_config.ProjectConfig
            Project configuration; must have ``ccm_config.max_libsize``/
            ``ccm_config.libsize_step`` (and optionally
            ``ccm_config.min_window``/``ccm_config.min_libsize``), and
            ``output.csv.dir_structure``/``output.csv.file_format`` (or an
            ``intermediate.csv`` block — see :meth:`set_output_calc_sub`).
        proj_dir : str or pathlib.Path, optional
            Root project directory, passed to :class:`CMConfigBase`.
        cpus : int, optional
            Stored as ``self.cpus``. Default is ``1``.
        exclusion_radius : float, optional
            Passed to :class:`CMConfigBase`.
        limit_surr_libsizes : bool, optional
            If true (default), trims ``self.libsizes`` to its last 5 values
            when either variable is using surrogate data.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        rc = RunConfig(grp_specs)
        super().__init__(grp_specs, config, proj_dir=proj_dir, exclusion_radius=exclusion_radius, init_var_objs=True)

        self.file_name = self.get_filename(config)

        self.df = None
        # @TODO check weighting specs
        self.weighted = grp_specs['weighted'] if 'weighted' in grp_specs.keys() else False
        self.self_predict = grp_specs['self_predict'] if 'self_predict' in grp_specs.keys() else False
        self.overwrite = None
        try:
            self.min_window = config.ccm_config.min_window
        except:
            self.min_window  = None
        self.max_libsize = config.ccm_config.max_libsize
        self.libsize_step = config.ccm_config.libsize_step
        self.libsizes = None# np.arange(self.knn+1, self.max_libsize+1, self.libsize_step)


        self.calc_location = set_calc_path(None, self.proj_dir, config, second_suffix='')
        self.output_dir = set_output_path(None, self.calc_location, config)
        self.output_path = self.set_output_calc_sub(config, self.output_dir, self.file_name)
        self.file_path = self.output_path / self.file_name
        self.pred_num = None
        self.cpus = cpus
        self.embedded = False
        self.id_num = None


        self.set_col_ts()
        self.set_target_ts()

        self.make_df().shift()
        try:
            self.min_libsize = config.ccm_config.min_libsize
        except:
             self.min_libsize = self.knn + 5

        self.set_libsizes()

        if self.target_var_obj.ts_type == 'surr' or self.col_var_obj.ts_type == 'surr':
            if limit_surr_libsizes is True:
                self.libsizes = self.libsizes[-5:]

        extra_cols = [col for col in self.df.columns if col not in (self.col_var_obj.col_name, self.target_var_obj.col_name)]
        self.time_var = extra_cols[0] if len(extra_cols) >0 else None
        self.noTime = True if self.time_var is None else False

        if self.target_var_obj.ts_type == 'surr' or self.col_var_obj.ts_type == 'surr':
            self.sample = 100
        else:
            self.sample = 250

        self.rc = rc
        self.outputgrp = None

        # print('ccm config initialized with output path:', self.file_path)
        log_line(self.log, ['ccm config initialized with output path:', self.file_path],
                 indent=0, log_type="info")

    def get_filename(self, config):
        """Generate this run's output CSV filename from ``config``'s template, or a fallback pattern.

        Tries ``template_replace(config.output.csv.file_format, self.to_dict())``
        first; falls back to a hardcoded
        ``{pset_id}_E{E}_tau{tau}__{surr_var}{surr_num}.csv`` pattern if that
        fails for any reason (the ``except`` here is unqualified, so it also
        catches errors unrelated to the template lookup itself).

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig

        Returns
        -------
        str
            Filename, normalized via :func:`check_csv`.
        """
        # generate filename of CCM CSV based on template in config
        pset_d = self.to_dict()
        try:
            file_name_template = config.output.csv.file_format
            file_name = template_replace(file_name_template, pset_d, return_replaced=False)# f'{replace(file_name_template, pset_d)}.csv'
        except:
            file_name = f"{pset_d['pset_id']}_E{pset_d['E']}_tau{pset_d['tau']}__{pset_d['surr_var']}{pset_d['surr_num']}.csv"

        return check_csv(file_name)

    def check_run_exists(self):
        """Check whether this run's output file (or its stem) already exists on disk.

        Returns
        -------
        tuple[bool, bool] or bool
            ``(pset_exists, stem_exists)`` from :func:`check_exists`, where
            ``pset_exists`` is the strong existence criterion (exact file)
            and ``stem_exists`` is the looser one (matching stem).

        Note
        ----
        The body has a ``self.output_path is None or self.file_name is
        None`` guard that returns ``False`` early — but it's written
        *after* the ``check_exists(check_csv(self.file_name),
        Path(self.output_path))`` call, not before. ``Path(None)`` raises
        ``TypeError``, so as written this guard can never actually run if
        either value is ``None`` — the call above it would already have
        raised. See ``notes_core_todos.md`` for follow-up; not changed here.
        """
        pset_exists, stem_exists = check_exists(check_csv(self.file_name), Path(self.output_path))
        if self.output_path is None or self.file_name is None:
            return False

        if pset_exists != self.file_path.exists():
            print(f'Warning: mismatch between expected existence {pset_exists} and actual existence {self.file_path.exists()} for {self.file_path}')

        print(f'Checking existence of CCM output at {self.file_path}: {pset_exists}')
        return pset_exists, stem_exists

    def set_output_calc_sub(self, config, output_dir, file_name):
        """Resolve this run's output subdirectory under ``output_dir`` (or an ``intermediate`` location).

        If ``config`` is in CSV entry mode (``config.get_entry() ==
        "csv"``) and has an ``intermediate.csv`` block, routes output under
        ``self.calc_location / "intermediate" / <csv dir>`` instead of the
        normal output tree, filling in ``intermediate.csv.dir_structure``.
        Otherwise fills in ``config.output.csv.dir_structure`` under
        ``output_dir``.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig
        output_dir : str or pathlib.Path
            Base output directory used in the non-intermediate branch.
        file_name : str
            Currently unused by this method's body.

        Returns
        -------
        pathlib.Path
        """
        # Route raw CCM CSV outputs to intermediate when configured.
        use_intermediate = getattr(config, "get_entry", lambda: None)() == "csv" and hasattr(config, "intermediate")
        if use_intermediate and hasattr(config.intermediate, "csv"):
            csv_block = config.intermediate.csv
            base = Path(self.calc_location) / "intermediate" / (getattr(csv_block, "dir", "csv") or "csv")
            grp_path_template = getattr(csv_block, "dir_structure", "")
            grp_path_template_filled = template_replace(grp_path_template, self.to_dict(), return_replaced=False)
            grp_path = base / grp_path_template_filled if grp_path_template_filled else base
        else:
            grp_path_template = config.output.csv.dir_structure#config.get_dynamic_attr("output.{var}", 'dir_structure_csv')  # config.output.grp_dir_structure
            grp_path_template_filled = template_replace(grp_path_template, self.to_dict(), return_replaced=False)
            grp_path = self.output_dir / grp_path_template_filled

        return grp_path

    def set_libsizes(self):
        """Set ``self.libsizes`` to the libsize values to sweep over.

        If ``self.min_window`` is set, sweeps two bands — near
        ``self.min_libsize`` and near ``self.max_libsize``, each
        ``self.min_window`` wide — rather than the full range between them.
        Otherwise sweeps the full range
        ``[self.min_libsize, self.max_libsize]`` at ``self.libsize_step``.
        """
        if self.min_window is not None:
            self.libsizes = np.concatenate([np.arange(self.min_libsize, self.min_libsize+self.min_window, self.libsize_step),
                                            np.arange(self.max_libsize -self.min_window, self.max_libsize, self.libsize_step)])
            log_line(self.log, ['running reduced libsize spread: ', self.libsizes],indent=1, log_type="info")
        else:
            self.libsizes = np.arange(self.min_libsize, self.max_libsize + 1, self.libsize_step)


    # def set_col_ts(self, surr_num=None):
    #     # DUPLICATE of CMConfigBase.set_col_ts (identical body, minus that one's
    #     # `col_var_obj is None` guard, which is never relevant here since CCMConfig.__init__
    #     # always resolves it first). Deletion candidate — see notes_core_todos.md.
    #     if self.col_var_obj.ts_type == 'surr':
    #         if (self.col_var_obj.surr_num is None) and (surr_num is not None):
    #             self.col_var_obj.surr_num = surr_num
    #
    #     if self.col_var_obj.surr_num not in (0, None):
    #         self.col_var_obj.get_surr(self.col_var_obj.surr_num)
    #     else:
    #         self.col_var_obj.get_real()

    def set_target_ts(self, surr_num=None):
        # NOT a pure duplicate of CMConfigBase.set_target_ts — behaves differently when
        # self.surr_var matches the target side but surr_num is None: this override forces
        # self.target_var_obj.surr_num = 0 (real data) in that case, while the base class
        # version leaves the existing surr_num untouched. Do not delete without checking
        # which behavior is intended — see notes_core_todos.md.
        if self.surr_var in ('y', self.target_var, 'both'):
            if (self.target_var_obj.surr_num is None) and (surr_num is not None):
                self.target_var_obj.surr_num = surr_num
            else:
                self.target_var_obj.surr_num = 0

        if self.target_var_obj.surr_num not in (0, None):
            self.target_var_obj.get_surr(self.target_var_obj.surr_num)
        else:
            self.target_var_obj.get_real()

    def make_df(self):
        # DUPLICATE of CMConfigBase.make_df (identical body, plus a chunk of dead
        # commented-out code below). Deletion candidate — see notes_core_todos.md.
        self.df = merge_variable_ts(self.col_var_obj, self.target_var_obj)
        # col_df = self.col_var_obj.ts.rename(columns={self.col_var_obj.col_name: self.col_var_obj.var})
        # target_df = self.target_var_obj.ts.rename(columns={self.target_var_obj.col_name: self.target_var_obj.var})
        # try:
        #     merged_df = pd.merge(col_df, target_df, on=self.col_var_obj.time_var, how='inner')
        # except:
        #     time_types = [type(self.col_var_obj.delta_ts), type(self.target_var_obj.delta_ts)]
        #     if any([t in [int, float, np.int64, np.float64] for t in time_types]):
        #         col_df[self.col_var_obj.time_var] = col_df[self.col_var_obj.time_var].astype(float)
        #         target_df[self.target_var_obj.time_var] = target_df[self.target_var_obj.time_var].astype(float)
        #     else:
        #         col_df[self.col_var_obj.time_var] = col_df[self.col_var_obj.time_var].astype(int)
        #         target_df[self.target_var_obj.time_var] = target_df[self.target_var_obj.time_var].astype(int)
        #     merged_df = pd.merge(col_df, target_df, on=self.col_var_obj.time_var, how='inner')
        #
        # self.df = merged_df.sort_values(by=self.col_var_obj.time_var).reset_index(drop=True)

        # self.train_ind_f = self.df.index.values[-1] if self.train_ind_f is None else self.train_ind_f
        self.df = self.df.iloc[self.train_ind_i : self.train_ind_f].reset_index(drop=True) if self.train_ind_f is not None else self.df.iloc[self.train_ind_i : ].reset_index(drop=True)
        return self

    def shift(self):
        """Shift ``self.target_var`` by ``self.lag``, drop the resulting NaNs, and trim ``max_libsize`` to fit.

        Mutates ``self.df`` in place (replacing it with the shifted,
        NaN-dropped, reindexed version), updates ``self.train_ind_f`` to the
        new last index, and caps ``self.max_libsize`` at 75% of the
        resulting row count if it was larger.
        """
        shifted = self.df.copy()
        shifted[self.target_var] = shifted[self.target_var].shift(self.lag)
        shifted = shifted.dropna()

        self.train_ind_f = shifted.index.values[-1] #if self.train_ind_f is None else self.train_ind_f

        self.df = shifted.reset_index(drop=True)
        self.max_libsize = min(self.max_libsize, int(.75*len(self.df)))

    def run_ccm(self, overwrite=None, ind=None, args=None, script=None):
        """Run this CCM configuration via ``run_experiment`` and store the result as an ``OutputCollection``.

        Normalizes ``args`` to a ``SimpleNamespace`` (building a default
        one from ``overwrite`` if not given), checks whether this run's
        output already exists (:meth:`check_run_exists`), determines
        overwrite/continue behavior via
        ``cedarkit.utils.io.gonogo.decide_file_handling``, runs
        ``cedarkit.utils.experiments.run_experiment``, writes the result via
        ``write_to_file``, and wraps it in ``self.outputgrp``.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite an existing output file. Ignored if
            ``args`` is given (use ``args.override`` instead).
        ind : int, optional
            If given, stored as ``self.id_num`` and passed through to
            ``run_experiment``.
        args : SimpleNamespace or dict or object, optional
            Run arguments; coerced to a ``SimpleNamespace`` if not already
            one. Must end up with ``override``/``write`` (and
            ``datetime_flag``) attributes.
        script : Any, optional
            Passed through to ``run_experiment``.

        Returns
        -------
        tuple
            ``(ccm_out_df, df_path)`` from ``run_experiment``.

        Raises
        ------
        TypeError
            If ``args`` is given but isn't a ``SimpleNamespace``, ``dict``,
            or an object with a ``__dict__``.

        Note
        ----
        ``self.check_run_exists()`` is called twice in a row (the first
        result is used only to compute ``overwrite_flag``, which is then
        never read again — ``overwrite`` is reassigned from
        ``decide_file_handling``'s return value instead). Separately,
        ``decide_file_handling``'s ``run_continue`` return value is also
        never read after being assigned — the method proceeds to call
        ``run_experiment``/``write_to_file`` unconditionally regardless of
        what it says. Whether this is intentional (e.g. ``write_to_file``
        itself decides what to do with ``overwrite``, making
        ``run_continue`` advisory/logging-only) isn't visible from this
        method alone — see ``notes_core_todos.md``.
        """
        from cedarkit.utils.experiments import run_experiment, write_to_file
        from cedarkit.utils.io.gonogo import decide_file_handling
        if ind is not None:
            self.id_num = ind


        if args is None:
            if overwrite is None:
                overwrite = False
            args = SimpleNamespace(override=overwrite, datetime_flag=None, write='append')
        else:
            if not isinstance(args, SimpleNamespace):
                if isinstance(args, dict):
                    args = SimpleNamespace(**args)
                elif hasattr(args, '__dict__'):
                    args = SimpleNamespace(**vars(args))
                else:
                    raise TypeError(f'Unsupported args type for run_ccm: {type(args)}')
            print('args provided for CCM run:', args, file=sys.stderr, flush=True)

        pset_exists, stem_exists = self.check_run_exists()
        overwrite_flag = overwrite if overwrite is not None else self.overwrite

        pset_exists, stem_exists = self.check_run_exists()
        # this is strong existence criteria... if want to check for stem existence, use stem_exists

        run_continue, overwrite = decide_file_handling(args, pset_exists)
        print(f'CCM run file handling decision - pset_exists: {pset_exists}, overwrite: {overwrite}, run_continue: {run_continue}', file=sys.stderr, flush=True)

        ccm_out_df, df_path = run_experiment((self, script, ind))

        write_to_file(ccm_out_df, df_path, overwrite=overwrite)
        self.outputgrp = OutputCollection(grp_specs= self.rc, in_table=ccm_out_df, tmp_dir=self.proj_dir/'tmp')

        return ccm_out_df, df_path
