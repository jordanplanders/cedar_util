"""Per-variable config resolution and timeseries loading.

``DataVarConfig`` resolves a single variable's entry in a
:class:`~cedarkit.core.project_config.ProjectConfig` into concrete file
locations, column names, and display metadata for its real and surrogate
timeseries. ``VarObject`` subclasses it to additionally load that data from
disk and expose it as a ``pandas.DataFrame`` (``.ts``) or a
``pyleoclim.Series`` (``.ps``).
"""

from pathlib import Path
import pandas as pd
import numpy as np

import sys
import logging
import re
import unicodedata
logger = logging.getLogger(__name__)
# import cedarkit.utils.routing.paths

# from utils.data_access import choose_data_source, check_csv, remove_extra_index

try:
    from cedarkit.utils.routing.file_name_parsers import check_csv
    from cedarkit.utils.routing.paths import check_location
    from cedarkit.utils.io.timeseries_utils import choose_data_source, remove_extra_index
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.routing.file_name_parsers import check_csv
    from utils.routing.paths import check_location
    from utils.io.timeseries_utils import choose_data_source, remove_extra_index
    from utils.cli.logging import setup_logging, log_line


class DataVarConfig:
    """Resolved config for a single data variable.

    Looks up ``var_id`` in ``config``, resolves its real and surrogate
    timeseries source info (CSV stem, value column, time column) into
    concrete file paths and column names, and pulls in display metadata
    (``color``, ``unit``, ``var_label``, etc.). Resolution happens
    immediately in :meth:`__init__` via :meth:`populate` — the instance is
    fully resolved as soon as it is constructed.
    """

    def __init__(self, config, var_id, proj_dir, suffix_label=None, suffix_ind=None):
        """Resolve ``var_id``'s entry in ``config`` and populate this instance.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig or None
            Project configuration. Must have a top-level attribute named
            ``var_id`` holding that variable's config block (with optional
            nested ``real_data_ts``/``surrogate_ts`` blocks), and a ``pal``
            attribute holding the variable-id-to-color palette. When ``None``,
            load the variable's project-level config from ``proj_dir``.
        var_id : str
            Key identifying this variable in ``config``.
        proj_dir : str or pathlib.Path
            Root directory of the project, used to resolve relative data paths.
        suffix_label : str, optional
            Label component appended (with ``suffix_ind``) to CSV stems and
            column names to disambiguate variants of the same variable.
            Default is ``''``.
        suffix_ind : str, optional
            Index/run component appended to CSV stems and column names.
            Default is ``''``.
        """
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.var_id = var_id
        self.var = None  # e.g. 'temp'

        self.suffix_label = suffix_label if suffix_label is not None else ''
        self.suffix_ind = suffix_ind if suffix_ind is not None else ''
        self.suffix = f'{self.suffix_label}{self.suffix_ind}'

        self.real_ts_var = None
        self.real_ts_time = None

        self.surr_ts_time = None
        self.surr_ts_var = None
        self.surr_csv_stem = None
        self.surr_prefix = None

        # self.surr_csvs = None
        # self.surr_time_var = None
        # self.surr_var = None  # e.g. 'temp'
        # self.surr_num = None

        self.obs_type = None
        self.source = None
        self.unit = None
        self.time_unit = None
        self.delta_t = None
        self.var_label = None
        self.var_name = None
        self.year = None
        self.color = None
        self.author = None

        # TODO there is some future redundancy here and sketchy path handling
        self.real_data_dir_path = None
        self.surr_data_dir_path = None
        self.proj_dir = None

        # self.real_ts_col = None
        self.real_ts_csv = None
        self.surr_ts_csv = None

        self.populate(config, proj_dir)

    def populate(self, config, proj_dir):
        """Resolve real/surrogate source info for ``self.var_id`` from ``config``.

        Looks up ``self.var_id`` in ``config`` (via
        ``config.get_dynamic_attr("{var}", self.var_id)``) and converts it to
        a plain dict, ``var_info``, whose nested ``real_data_ts`` and
        ``surrogate_ts`` blocks become ``real_ts_d``/``surr_ts_d``.

        For each of the CSV stem, value column, and time column (real and
        surrogate), several possible key names are tried in priority order
        and the first one present wins:

        - real CSV stem: ``var_info['real_csv_stem']`` ->
          ``real_ts_d['csv_stem']`` -> ``var_info['data_csv']``
        - real value column: ``var_info['real_ts_var']`` ->
          ``real_ts_d['var']`` -> ``var_info['data_var']``
        - real time column: ``var_info['real_ts_time']`` ->
          ``real_ts_d['real_time_var']`` -> ``real_ts_d['time']`` ->
          ``var_info['time_var']`` -> ``'time'``
        - surrogate CSV stem: ``var_info['surr_csv_stem']`` ->
          ``surr_ts_d['csv_stem']`` -> ``var_info['surr_file_name']``
          (with ``'.txt'`` stripped)
        - surrogate value column: ``var_info['surr_var']`` ->
          ``surr_ts_d['var']`` -> ``self.var``
        - surrogate time column: ``var_info['surr_time_var']`` ->
          ``surr_ts_d['time']`` -> ``'date'``

        If none of a value's candidate keys are present, resolution falls
        back to the listed default (or, for the two CSV stems, a message is
        printed and the attribute is left unset) rather than raising.

        After resolving the above, the real and surrogate CSV filenames and
        data directory paths are computed, the palette color is resolved via
        :meth:`get_color`, and any remaining keys in ``var_info`` that match
        an existing (still-``None``) attribute name are copied onto ``self``.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig or None
            Project configuration containing this variable's config block. When
            ``None``, load ``<proj_dir>/data_var_configs/<var_id>.yaml`` as a
            single-variable project config and use the project's master data
            directories instead of dyad routing.
        proj_dir : str or pathlib.Path
            Root directory of the project, used to resolve relative data paths.
        """
        self.proj_dir = proj_dir
        using_master_config = config is None
        if using_master_config:
            if proj_dir is None:
                raise ValueError("proj_dir is required when config is None.")
            from cedarkit.core.project_config import load_config

            config_path = Path(proj_dir) / 'data_var_configs' / f'{self.var_id}.yaml'
            config = load_config(config_path)
        # try:
        #     var_yaml = config.get_dynamic_attr("data_vars.{var}", self.var_id)
        #     var_info = load_config(proj_dir / 'var_configs'/f'{var_yaml}.yaml')
        #
        #     # load variable-specific settings from config
        #     # self.load_from_var_yaml(var_yaml, proj_dir)
        #     var_info = var_yaml.get(self.var_id, None) if var_yaml is not None else None
        # except:
        # print(f'reading var yaml for {self.var_id} failed, trying config')
        # print(self.var_id)

        var_info = config.get_dynamic_attr("{var}", self.var_id)
        var_info = var_info.to_dict()
        # self.load_from_config(config, proj_dir)

        # set real data info
        real_ts_d = var_info.pop('real_data_ts', {}) or {}
        # real_csv_stem
        if 'real_csv_stem' in var_info.keys():
            self.real_csv_stem = var_info.pop('real_csv_stem', None)
        elif 'csv_stem' in real_ts_d.keys():
            self.real_csv_stem = real_ts_d.pop('csv_stem', None)
        elif 'data_csv' in var_info.keys():
            self.real_csv_stem = var_info.pop('data_csv', None)
        else:
            base_var = var_info.get('var') or self.var_id
            stem_fields = [
                var_info.get('author'), var_info.get('year'),
                var_info.get('source'), var_info.get('obs_type'),
            ]
            if all(field not in (None, '') for field in stem_fields):
                self.real_csv_stem = '_'.join([
                    *(self._stem_token(field) for field in stem_fields),
                    'decavg', self._stem_token(base_var),
                ])
            else:
                self.real_csv_stem = f'{self.var_id}_decavg_{base_var}'

        # real_ts_var
        if 'real_ts_var' in var_info.keys():
            self.real_ts_var = var_info.pop('real_ts_var', None)
        elif 'var' in real_ts_d.keys():
            self.real_ts_var = real_ts_d.pop('var', None)
        elif 'data_var' in var_info.keys():
            self.real_ts_var = var_info.pop('data_var', None)
        else:
            base_var = var_info.get('var') or self.var_id
            self.real_ts_var = f'{self.var_id}_decavg_{base_var}'

        # real_ts_time
        if 'real_ts_time' in var_info.keys():
            self.real_ts_time = var_info.pop('real_ts_time', None)
        elif 'real_time_var' in real_ts_d.keys():
            self.real_ts_time = real_ts_d.pop('real_time_var', None)
        elif 'time' in real_ts_d.keys():
            self.real_ts_time = real_ts_d.pop('time', None)
        elif 'time_var' in var_info.keys():
            self.real_ts_time = var_info.pop('time_var', None)
        else:
            self.real_ts_time = 'time'

        self.set_real_csv_name()
        if using_master_config:
            self.real_data_dir_path = Path(proj_dir) / 'master_data'
        else:
            self.real_data_dir_path = self.set_data_source(config, data_source='data', data_type='real')
        self.get_color(config)

        surr_ts_d = var_info.pop('surrogate_ts', {}) or {}
        # print(var_info, surr_ts_d)
        #surr_csv_stem
        if 'surr_csv_stem' in var_info.keys():
            self.surr_csv_stem = var_info.pop('surr_csv_stem', None)
        elif 'csv_stem' in surr_ts_d.keys():
            self.surr_csv_stem = surr_ts_d.pop('csv_stem', None)
        elif 'surr_file_name' in var_info.keys():
            surr_file_name = var_info.pop('surr_file_name', None)
            self.surr_csv_stem = surr_file_name.replace('.txt', '')
        else:
            print(f'No surr_csv_stem found for {self.var_id}')

        # surr_ts_var
        if 'surr_var' in var_info.keys():
            self.surr_ts_var = var_info.pop('surr_var', None)
        elif 'var' in surr_ts_d.keys():
            self.surr_ts_var = surr_ts_d.pop('var', None)
        else:
            self.surr_ts_var = self.var
        self.surr_prefix = self.surr_ts_var

        # surr_ts_time
        if 'surr_time_var' in var_info.keys():
            self.surr_ts_time = var_info.pop('surr_time_var', None)
        elif 'time' in surr_ts_d.keys():
            self.surr_ts_time = surr_ts_d.pop('time', None)
        else:
            self.surr_ts_time = 'date'

        self.set_surr_csv_name()
        if self.surr_ts_csv is not None:
            if using_master_config:
                self.surr_data_dir_path = Path(proj_dir) / 'master_surrogates'
            else:
                self.surr_data_dir_path = self.set_data_source(config, data_source='data', data_type='surr')

        for key in var_info.keys():
            if hasattr(self, key):
                if getattr(self, key) is None:
                    try:
                        setattr(self, key, var_info[key])
                    except Exception as e:
                        print(f'Error setting attribute {key} for var_id {self.var_id}: {e}')

    def set_surr_csv_name(self):
        # Mutator: sets self.surr_ts_csv from surr_csv_stem + suffix. No return value.
        if len(self.suffix) > 0:
            self.surr_ts_csv = '__'.join([self.surr_csv_stem, self.suffix]).strip(
                '__') if self.surr_csv_stem is not None else None
        else:
            if (self.surr_csv_stem is not None) and (self.surr_csv_stem !=''):
                self.surr_ts_csv = self.surr_csv_stem
            else:
                self.surr_ts_csv = None

    def set_real_csv_name(self):
        # Mutator: sets self.real_ts_csv from self.real_csv_stem. No return value.
        if len(self.suffix) > 0:
            self.real_ts_csv = '__'.join([self.real_csv_stem, self.suffix]).strip(
                '__') if self.real_csv_stem is not None else None
        else:
            self.real_ts_csv = self.real_csv_stem

    @staticmethod
    def _stem_token(value):
        """Convert metadata to a portable underscore-delimited filename token."""
        normalized = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode()
        return re.sub(r'[^A-Za-z0-9]+', '_', normalized).strip('_')

    def set_data_source(self, config, data_source='data', var_data_csv=None, data_type='real'):
        """Resolve the directory containing this variable's data CSV.

        First tries ``choose_data_source`` to find ``var_data_csv`` under
        ``proj_dir / data_source``. If that doesn't resolve a path, falls
        back to a location-specific config directory (``raw_data_dir`` /
        ``raw_data`` for real data, ``surr_data_dir`` / ``surrogate_data``
        for surrogate data, each looked up via
        ``config.get_dynamic_attr('{var}', check_location(self.proj_dir))``),
        and finally to a hardcoded default subdirectory name (``'data'`` or
        ``'surrogates'``) if even that lookup fails.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig
            Project configuration, used for the location-specific fallback.
        data_source : str, optional
            Subdirectory of ``proj_dir`` to search first. Default is ``'data'``.
        var_data_csv : str, optional
            CSV stem to look for. If not given, defaults to
            ``self.real_ts_csv`` or ``self.surr_ts_csv`` depending on
            ``data_type``.
        data_type : {'real', 'surr', 'surrogate'}, optional
            Which of the variable's two timeseries this resolves a directory
            for. Default is ``'real'``.

        Returns
        -------
        pathlib.Path
            Directory expected to contain the variable's data CSV.
        """
        if var_data_csv is None:
            if data_type == 'real':
                var_data_csv = self.real_ts_csv
            elif data_type in ['surr', 'surrogate']:
                var_data_csv = self.surr_ts_csv

        data_path, _ = choose_data_source(self.proj_dir, config, data_source, data_type=data_type,
                                          var_data_csv=var_data_csv)
        if data_path is None:
            try:
                loc_config = config.get_dynamic_attr('{var}', check_location(self.proj_dir))
            except Exception:
                loc_config = None
            if data_type == 'real':
                config_source = getattr(loc_config, 'raw_data_dir', None) if loc_config is not None else None
                if config_source is None:
                    config_source = getattr(loc_config, 'raw_data', None) if loc_config is not None else None
                config_source = config_source if config_source is not None else 'data'
            elif data_type in ['surr', 'surrogate']:
                config_source = getattr(loc_config, 'surr_data_dir', None) if loc_config is not None else None
                if config_source is None:
                    config_source = getattr(loc_config, 'surrogate_data', None) if loc_config is not None else None
                config_source = config_source if config_source is not None else 'surrogates'
            else:
                config_source = 'data'
            return Path(self.proj_dir) / config_source
        data_path = Path(data_path).parent
        return data_path

    def get_color(self, config):
        # Mutator: sets self.color from config.pal if unset (defaults to 'black'). No return value.
        if self.color is None:
            color_map = config.pal.to_dict()
            if color_map is not None and self.var_id in color_map:
                self.color = color_map[self.var_id]
            else:
                self.color = 'black'


class VarObject(DataVarConfig):
    """A resolved data variable plus its loaded timeseries.

    Extends :class:`DataVarConfig` with the actual timeseries data (``ts``)
    and the bookkeeping needed to load either the real series or one of its
    surrogates (``ts_type``, ``surr_num``, ``col_name``, ``time_var``).

    Can be constructed two ways: by resolving ``config``/``var_id``/``proj_dir``
    from scratch (same as ``DataVarConfig``), or by copying an
    already-resolved ``DataVarConfig`` instance via ``data_var_config`` to
    avoid re-resolving it. See :meth:`__init__`.
    """

    def __init__(self, config, var_id=None, proj_dir=None, data_var_config=None,
                 suffix_label=None, suffix_ind=None):
        """Construct a ``VarObject``, either fresh or from an existing config.

        If ``data_var_config`` is given, its attributes (other than ``log``)
        are copied onto this instance and ``config``/``var_id``/``proj_dir``
        are ignored. Otherwise, ``config``/``var_id``/``proj_dir`` are passed
        to :class:`DataVarConfig`'s constructor to resolve a new config.

        Parameters
        ----------
        config : cedarkit.core.project_config.ProjectConfig or None
            Project configuration. Only used when ``data_var_config`` is
            ``None``. When it is ``None``, the project-level variable config
            is loaded from ``proj_dir``.
        var_id : str, optional
            Key identifying this variable in ``config``. Only used when
            ``data_var_config`` is ``None``.
        proj_dir : str or pathlib.Path, optional
            Root directory of the project. Only used when ``data_var_config``
            is ``None``.
        data_var_config : DataVarConfig, optional
            An already-resolved config to copy instead of resolving a new one.
        suffix_label, suffix_ind : str, optional
            Components appended after ``'__'`` to the resolved real CSV and
            real value-column names when constructing from a config.
        """
        if data_var_config is not None and isinstance(data_var_config, DataVarConfig):
            # Copy all attributes from the provided DataVarConfig
            for key, value in data_var_config.__dict__.items():
                if key !='log':
                    setattr(self, key, value)
        else:
            # Initialize as a new DataVarConfig
            super().__init__(config, var_id, proj_dir,
                             suffix_label=suffix_label, suffix_ind=suffix_ind)

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.ts = None
        self.ts_type = None  # 'real' or 'surr'
        self.surr_num = None
        self.col_name = None
        self.time_var = None

        # self.pyleo_ts = None

    @property
    def ps(self):
        """This variable's currently-loaded timeseries as a pyleoclim Series.

        Requires ``self.ts`` to already be populated (via :meth:`pull_ts`,
        :meth:`get_real`, or :meth:`get_surr`) — built directly from
        ``self.ts``, ``self.time_var``, and ``self.col_name``.

        Returns
        -------
        pyleoclim.Series
            Series with ``time_unit`` defaulting to ``'yr BP'`` if
            ``self.time_unit`` is unset, and ``value_unit``/``value_name``/
            ``label`` taken from ``self.unit``/``self.var``/``self.var_name``.
        """
        import pyleoclim as pyleo

        time_axis = np.abs(self.ts[self.time_var].values) # absolute time values for pyleo, imply direction via time_unit
        source_ps = pyleo.Series(time=time_axis, value=self.ts[self.col_name].values,
                                 time_unit=self.time_unit if self.time_unit is not None else 'yr BP', value_unit=self.unit, value_name=self.var,
                                 # label='wu_tsi')
                                 label=self.var_name)
        return source_ps

    # @property
    # def surrogate_ts(self):
    #     return self.get_surr(surr_num=self.surr_num)

    def set_col_name(self):
        # Mutator: sets self.col_name from self.ts_type ('real' or 'surr'). No return value.
        if self.ts_type == 'real':
            if len(self.suffix) > 0:
                self.col_name = '__'.join([self.real_ts_var, self.suffix]).strip(
                    '__') if self.real_ts_var is not None else None
            else:
                self.col_name = self.real_ts_var

            # self.col_name = self.raw_data_col
        elif self.ts_type == 'surr':
            self.col_name = f'{self.surr_prefix}_{self.surr_num}'

    def standardize_time_var(self, specified_time_var, df, other_col):
        """Rename ``df``'s time column to ``'time'`` and infer ``self.delta_t``.

        If ``df`` has no ``'time'`` column, renames ``specified_time_var`` (or
        ``'date'``, if present) to ``'time'``. If ``self.delta_t`` is unset,
        infers it from the minimum spacing between sorted, unique time
        values: if at least 90% of the diffs are effectively integer-valued,
        ``delta_t`` is set to the rounded minimum as an ``int``; otherwise the
        raw minimum is kept as a ``float``.

        Parameters
        ----------
        specified_time_var : str or None
            Name of the time column to rename to ``'time'``, if ``df``
            doesn't already have one.
        df : pandas.DataFrame
            Data to standardize. Mutated in place (column rename) and returned.
        other_col : str
            Currently unused by this method's body — accepted but not
            referenced; appears to be unused/vestigial rather than load-bearing.

        Returns
        -------
        tuple[pandas.DataFrame, str]
            The (possibly renamed) ``df`` and the literal string ``'time'``.
        """

        if ('time' not in df.columns) and (specified_time_var is not None):
            df = df.rename(columns={specified_time_var: 'time'})
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})

        if self.delta_t is None:
            # infer delta_t from time values
            time_diffs = df['time'].sort_values().diff().dropna().unique()
            if len(time_diffs) > 0:
                td = np.asarray(time_diffs, dtype=float)
                # proportion of diffs that are effectively integer-valued
                is_int = np.isclose(td, np.round(td), atol=1e-8)
                int_ratio = np.mean(is_int)
                if int_ratio >= 0.9:
                    # mostly integers: use rounded minimum as integer
                    self.delta_t = int(np.min(np.round(td)))
                else:
                    # mostly floats: keep the minimum as float
                    self.delta_t = float(np.min(td))
            else:
                self.delta_t = None  # default to 1 if unable to infer            else:
        # self.delta_t = 1  # default to 1 if unable to infer
        # df['time'] = df['time'].astype('int')

        return df, 'time'


    def pull_ts(self, surr_num=None):
        # Dispatcher: calls get_real() if surr_num is None, else get_surr(surr_num). No return value.
        if surr_num is None:
            self.get_real()
        else:
            self.surr_num = surr_num
            self.get_surr(surr_num=surr_num)


    def get_real(self):
        # Mutator: loads the real CSV into self.ts (sets ts_type/col_name/time_var too).
        # Silent no-op if the expected CSV file doesn't exist. No return value.
        self.ts_type = 'real'
        self.set_col_name()

        if (self.real_data_dir_path / check_csv(self.real_ts_csv)).exists() is True:
            real_data = pd.read_csv(self.real_data_dir_path / check_csv(self.real_ts_csv))
            # print('raw data read', raw_data.head())
            real_data = remove_extra_index(real_data)
            # print('raw data before standardize', raw_data.head())

            real_data, time_var = self.standardize_time_var(self.real_ts_time, real_data, self.col_name)
            self.time_var = time_var
            # print('raw data', raw_data.head())

            self.ts = real_data[[self.time_var, self.col_name]].copy()

    def get_surr(self, surr_num=None):
        # Mutator: loads the surrogate CSV into self.ts (sets ts_type/col_name/time_var too).
        # Silent no-op if the expected CSV file doesn't exist. No return value.
        # print('sur', self.surr_data_dir_path / check_csv(self.surr_csv))
        if (self.surr_data_dir_path / check_csv(self.surr_ts_csv)).exists() is True:
            surr_data = pd.read_csv(self.surr_data_dir_path / check_csv(self.surr_ts_csv))
            surr_data = remove_extra_index(surr_data)
            # print(surr_data)

            # self.surr_num = self.surr_num if self.surr_num is not None else surr_num
            self.set_col_name()
            self.ts_type = 'surr'

            surr_data, time_var = self.standardize_time_var(self.real_ts_time, surr_data, self.col_name)
            self.time_var = time_var
            # print('surr data', surr_data[[self.time_var, self.col_name]].head())
            self.ts = surr_data[[self.time_var, self.col_name]].copy()
            # print(self.ts.head())
