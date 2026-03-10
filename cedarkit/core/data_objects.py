
import collections.abc
from copy import deepcopy
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from functools import reduce
import operator
from collections import defaultdict
import uuid
# from pyarrow import table
import gc
import pyarrow as pa
import pyarrow.compute as pc
from types import SimpleNamespace

# import cedarkit.utils.paths
# from cedarkit.utils.paths import set_calc_path, set_output_path, template_replace, check_exists

# from core.data_var import *
try:
    from cedarkit.core.data_var import *
    from cedarkit.core.relationship import *
    from cedarkit.utils.routing import *
    from cedarkit.utils.routing import template_replace
    from cedarkit.utils.tables import as_len1_array, as_lenN_array
    # from cedarkit.utils.cli.logging import print_log_line
    from cedarkit.utils.cli import log_line

except ImportError:
    # Fallback: imports when running as a package
    from core.data_var import *
    from core.relationship import *
    from utils.paths import *
    from utils.routing.file_name_parsers import template_replace
    from utils.tables.parquet_tools import _as_len1_array, _as_lenN_array
    from utils.cli.logging import log_line

# dump
import os
# SCRIPT = Path(__file__).resolve().name
import logging
logger = logging.getLogger(__name__)


def correct_iterable(obj):
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

# moved to data_access
#def template_replace(template, d, return_replaced=True):
#     replaced = []
#     old_template = template
#     for key, value in d.items():
#         template = cedarkit.utils.paths.replace(f'{{{key}}}', str(value))
#         if template != old_template:
#             replaced.append(key)
#             old_template = template
#     if return_replaced is False:
#         return template
#
#     return template, replaced
#

def extract_from_pattern(filename: str, pattern_str: str):
    """
    Extracts parameter values from filename based on a pattern string.
    Example:
        extract_from_pattern("E4_tau1_lag-5.parquet", "E{E}_tau{tau}_lag{lag}")
        -> {'E': 4, 'tau': 1, 'lag': -5}
    """
    # Convert format specifiers like {E}, {tau}, {lag} into named regex groups
    regex = re.sub(r"\{(\w+)\}",
                   lambda m: f"(?P<{m.group(1)}>-?\\d+)", pattern_str)

    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match pattern '{pattern_str}'")

    # Convert all extracted values to integers
    return {k: int(v) for k, v in match.groupdict().items()}

# def separate_real_surr(table):
#     if 'surr_var' in table.schema.names:
#         mask = pc.equal(table['surr_var'], 'neither')
#         real_table = table.filter(mask)
#         surr_table = table.filter(pc.invert(mask))
#         return real_table, surr_table
#     else:
#         return table

def check_return(table):
    if table is not None and table.num_rows > 0:
        return True
    else:
        return False


def compute_delta_rho_grp(
        lag_tbl: pa.Table,
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
    """
    Compute delta rho statistics and full vectors from lagged correlation table.

    Parameters
        - lag_tbl columns required: 'LibSize' (int/float), 'rho' (float)
        - gd: dict of group descriptors to copy into outputs
        - stats: whether to compute summary statistics table
        - full: whether to compute full vectors table

    Calculates
        - mean rho in min libsize band (libsize < min_libsize + min_window)
        - mean rho in max libsize band (libsize > max_libsize - max_window)
        - best libsize (argmax of mean rho by libsize)
        - mean rho in best libsize window (best_libsize +/- best_window_halfwidth)
        - delta rho = max libsize mean rho - min libsize mean rho
        - full vectors with bootstrap-style paired sampling (with replacement)

    Returns (stats_tbl | None, full_tbl | None) as pyarrow.Table objects.

    Used by OutputGrp.calc_delta_rho
    """
    # lag_tbl = self.table.full
    if lag_tbl is None or lag_tbl.num_rows == 0:
        log_line(logger, 'empty lag_tbl', indent=0,
                 log_type="info")
        # print_log_line(SCRIPT, inspect.currentframe().f_code.co_name, 'empty lag_tbl', level=0, log_type='error')
        return (None, None)

    lib = lag_tbl['LibSize']
    rho = lag_tbl['rho']

    # thresholds at ends
    lib_min = pc.min(lib).as_py()
    lib_max = pc.max(lib).as_py()

    # min/max libsize bands
    min_mask = pc.less(lib, lib_min + min_window)
    max_mask = pc.greater(lib, lib_max - max_window)

    min_tbl = lag_tbl.filter(min_mask)
    max_tbl = lag_tbl.filter(max_mask)

    gb = lag_tbl.group_by(["LibSize"]).aggregate([("rho", "mean")])  # columns: LibSize, rho_mean
    # sort by descending rho_mean
    gb_sorted = gb.sort_by([("rho_mean", "descending")])
    best_libsize = gb_sorted["LibSize"][0].as_py()

    # window around best libsize
    lo = best_libsize - best_window_halfwidth
    hi = best_libsize + best_window_halfwidth
    win_mask = pc.and_(
        pc.greater_equal(lib, lo),
        pc.less_equal(lib, hi)
    )
    best_tbl = lag_tbl.filter(win_mask)
    # stats
    stats_tbl = None

    n_min = min_tbl.num_rows
    n_max = max_tbl.num_rows
    n_best = best_tbl.num_rows
    sample_size = max(n_min, n_max)

    rng = np.random.default_rng(rng_seed)
    # sample indices with replacement from each subset
    idx_min = rng.integers(0, n_min, size=sample_size) if n_min > 0 else np.array([], dtype=np.int64)
    idx_max = rng.integers(0, n_max, size=sample_size) if n_max > 0 else np.array([], dtype=np.int64)
    idx_best = rng.integers(0, n_best, size=sample_size) if n_best > 0 else np.array([], dtype=np.int64)

    min_rhos = min_tbl['rho'].take(pa.array(idx_min)) if n_min > 0 else pa.array([], type=pa.float64())
    max_rhos = max_tbl['rho'].take(pa.array(idx_max)) if n_max > 0 else pa.array([], type=pa.float64())
    # align lengths (should already be sample_size)
    if len(min_rhos) != sample_size:
        min_rhos = pc.pad(min_rhos, target_length=sample_size)
    if len(max_rhos) != sample_size:
        max_rhos = pc.pad(max_rhos, target_length=sample_size)

    delta_rho_vec = pc.subtract(max_rhos, min_rhos)

    # also expose the raw rho values from the "best window"
    # best_rhos = best_tbl['rho'] if best_tbl.num_rows > 0 else pa.array([], type=pa.float64())
    best_rhos = best_tbl['rho'].take(pa.array(idx_best)) if n_min > 0 else pa.array([], type=pa.float64())

    if stats:
        best_mean_rho = pc.mean(best_tbl['rho']).as_py() if best_tbl.num_rows > 0 else np.nan
        min_mean_rho = pc.mean(min_tbl['rho']).as_py() if min_tbl.num_rows > 0 else np.nan
        max_mean_rho = pc.mean(max_tbl['rho']).as_py() if max_tbl.num_rows > 0 else np.nan
        delta_rho = (max_mean_rho - min_mean_rho) if (
                    np.isfinite(max_mean_rho) and np.isfinite(min_mean_rho)) else np.nan

        cols = {}
        # group descriptors (length-1 columns)
        for k, v in gd.items():
            cols[k] = as_len1_array(get_static(v))

        cols['maxrho'] = as_len1_array(best_mean_rho)# if np.isfinite(best_mean_rho) else np.nan)
        cols['minlibsize_rho'] = as_len1_array(min_mean_rho) #if np.isfinite(min_mean_rho) else np.nan)
        cols['maxlibsize_rho'] = as_len1_array(max_mean_rho)# if np.isfinite(max_mean_rho) else np.nan)
        cols['delta_rho'] = as_len1_array(delta_rho) #if np.isfinite(delta_rho) else np.nan)
        cols['annotation'] = as_len1_array(annotation)

        stats_tbl = pa.table(cols)

    # full vectors with bootstrap-style paired sampling (with replacement)
    full_tbl = None
    if full:

        cols_full = {
            'minlibsize_rho': min_rhos,
            'maxlibsize_rho': max_rhos,
            'delta_rho': delta_rho_vec,
            # For parity with your dict, expose maxrho as the vector from the best window
            'maxrho': best_rhos,
            'annotation': as_lenN_array(annotation, len(max_rhos)) #annotations#_repeat_scalar(annotation, sample_size, pa.string()),
        }
        # replicate gd for each row
        for k, v in gd.items():
            cols_full[k] = as_lenN_array(get_static(v), len(max_rhos))

        full_tbl = pa.table(cols_full)

    return stats_tbl, full_tbl


#########################################
class RunConfig:
    '''
    Configuration for a single CCM run, but can be extended to a group of runs
    grp_d: dictionary of group-level traits

    Methods:
        populate(grp_d): populate the RunConfig attributes from a dictionary
        copy(): create a deep copy of the RunConfig object
        to_dict(): convert the RunConfig attributes to a dictionary
        pull_output(to_table=False, limit_surr=True): pull output data based on the RunConfig attributes
        set_var_objs(proj_config, proj_dir): set variable objects for the RunConfig

    Inherited by DataGroup class
    '''
    def __init__(self, grp_d, tmp_dir=None, **_ignored_kwargs):

        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.E = None
        self.tau = None
        self.lag = None
        self.train_ind_i = 0
        self.train_ind_f = -1
        self.knn = None
        self.Tp = None
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
        # print('Populating RunConfig with grp_d:', grp_d, file=sys.stdout, flush=True)
        # print_log_line(SCRIPT, inspect.currentframe().f_code.co_name, ['Populating RunConfig with grp_d:', grp_d], level=1, log_type='info')
        log_line(
            self.log,
            ["Populating RunConfig with grp_d:", grp_d],
            indent=1,
            log_type="debug",  # or "info", but debug is nice for “comment/uncomment” style
        )

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
        return deepcopy(self)

    def get_trait_value(self, trait):
        return getattr(self, trait, None)

    @property
    def var_x(self):
        return self.col_var

    @property
    def var_y(self):
        return self.target_var

    @property
    def var_x_obj(self):
        return self.col_var_obj

    @property
    def var_y_obj(self):
        return self.target_var_obj

    @property
    def traits(self):
        return [key for key in self.__dict__.keys() if key not in ['output_path', 'log']]

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if key in self.traits and value is not None}

    def pull_output(self, to_table=False, limit_surr=True):
        if self.output_path is None or len(self.output_path) == 0:
            print('no output path specified')
            log_line(self.log, 'no output path specified', indent=0, log_type="error")
            return

        file_path = self.output_path[0]
        log_line(self.log, ['pulling from', file_path], indent=0, log_type="info")

        # print('pulling from', file_path)
        dset = ds.dataset(str(file_path), format="parquet")
        all_traits = self.to_dict()

        filters = {key: ds.field(key).isin(correct_iterable(value)) for key, value in all_traits.items() if
                   value is not None and key in dset.schema.names}
        combined_filter = reduce(operator.and_, filters.values())
        filtered_table = dset.to_table(filter=combined_filter)

        if to_table is True:
            return filtered_table
        else:
            return OutputCollection(in_table=filtered_table, grp_specs=self, outtype='full', tmp_dir=self.tmp_dir)

    def trait_hierarchy(self, full_ds, trait, level="below", threshold=0.9, include_ids=False):
        """
        Return traits that are above or below the grouping level of a given trait.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to analyze.
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
        if isinstance(full_ds, pa.Table):
            df = full_ds.to_pandas(types_mapper=pd.ArrowDtype)
        elif not isinstance(full_ds, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame or a pyarrow Table")
        else:
            df = full_ds

        if trait not in df.columns:
            raise ValueError(f"Trait '{trait}' not found in columns")

        grouped = df.groupby(trait)
        results = {}
        cols = df.columns if include_ids else [col for col in df.columns if ('id' not in col) and ('ind' not in col)]
        for col in self.traits:
            if col in df.columns:
                if col == trait:
                    continue
                # Uniqueness fraction within each group
                frac_unique = grouped[col].nunique(dropna=False) / grouped.size()
                results[col] = frac_unique.mean()

        if level == "below":
            return [col for col, frac in results.items() if frac > threshold]
        elif level == "above":
            return [col for col, frac in results.items() if frac <= threshold]
        else:
            raise ValueError("level must be 'below' or 'above'")

    def set_var_objs(self, proj_config, proj_dir):
        '''
        Set variable objects for the RunConfig based on project configuration.
        proj_config: ProjectConfig object containing project-level configurations
        proj_dir: Path object representing the project directory
        '''
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
    '''
    DataGroup object to manage a group of CCM runs based on shared traits.
    grp_d: dictionary of group-level traits
    Methods:
        get_files(config, output_path, file_name_pattern=None, source='parquet'): retrieve files matching the group traits
        pull_output(summary=True, full=False): pull output data from the group files

    Attributes:
        file_list: list of RunConfig objects for each file in the group
        grp_d: dictionary of group-level traits
        static_traits: dictionary of traits with single values
        nonstatic_traits: dictionary of traits with multiple values
        internal_traits: dictionary of traits determined during file retrieval
        parent_config: RunConfig object representing the group-level configuration
        output: OutputCollection object containing the pulled output data
        tmp_dir: temporary directory for intermediate files
        missing_files: dictionary of files that were expected but not found

    '''
    def __init__(self, grp_d, tmp_dir=None):
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


    # @TODO revise so checks for existence before returning relevant rows
    def _internal_query(self, dset, query_config=None):
        '''
        query_config: RunConfig object with specific values to filter on
        dset: pyarrow dataset object
        returns: (GroupConfig object, filtered pyarrow table)
        '''
        if query_config is not None:
            all_traits = query_config.to_dict()
        else:
            all_traits = self.grp_d#{**self.static_traits, **self.nonstatic_traits, **self.internal_traits}

        filters = {key: ds.field(key).isin(correct_iterable(value)) for key, value in all_traits.items() if
                   value is not None and key in dset.schema.names}

        combined_filter = reduce(operator.and_, filters.values())
        table = dset.to_table(filter=combined_filter)
        log_line(self.log, ['_internal_query: filtered table rows', table.num_rows], indent=0, log_type="debug")

        # print('_internal_query: filtered table rows', table.num_rows, file=sys.stdout, flush=True)
        # log_line(
        #     self.log,
        #     ["_internal_query: filtered table rows", table.num_rows],
        #     indent=0,
        #     log_type="debug",
        # )
        grp_info = {}
        for key in self.parent_config.traits:
            if key in table.schema.names:
                unique_elements = pc.unique(table[key]).to_pylist()
                grp_info[key] = unique_elements

        log_line(
            self.log,
            ["_internal_query: initial grp_info traits:", grp_info],
            indent=0,
            log_type="debug",
        )
        # print('_internal_query: initial grp_info traits:', grp_info, file=sys.stdout, flush=True)

        for key, value in all_traits.items():
            if value is None:
                continue
            if key in grp_info.keys():
                outliers = set(correct_iterable(value)) | set(grp_info[key])
            else:
                outliers = correct_iterable(value)
            if outliers is not None:
                grp_info[key] = correct_iterable(outliers)
        # print('\ttable info schema names:', table.schema.names, table.num_rows, file=sys.stdout, flush=True)
        # print('\t_internal_query: final grp_info traits:', grp_info, file=sys.stdout, flush=True)

        log_line(
            self.log,
            ["table info schema names:", table.schema.names, "rows:", table.num_rows],
            indent=1,
            log_type="debug",
        )
        log_line(
            self.log,
            ["_internal_query: final grp_info traits:", grp_info],
            indent=1,
            log_type="debug",
        )
        try:
            file_group_config = RunConfig(grp_info, tmp_dir=self.tmp_dir)
            # print('\t_internal_query: created RunConfig successfully', file=sys.stdout, flush=True)
            log_line(
                self.log,
                ["_internal_query: created RunConfig successfully"],
                indent=1,
                log_type="debug",
            )
        except Exception as e:
            # print('Failed to create RunConfig in _internal_query with grp_info:', grp_info, 'Error:', e, file=sys.stdout, flush=True)
            log_line(
                self.log,
                [
                    "Failed to create RunConfig in _internal_query with grp_info:",
                    grp_info,
                    "Error:",
                    e,
                ],
                indent=0,
                log_type="error",
            )
            raise e
        # print('\t_internal_query: RunConfig traits:', file_group_config, file=sys.stdout, flush=True)
        # print('_internal_query: returning table rows', table.num_rows, file=sys.stdout, flush=True)
        log_line(
            self.log,
            ["_internal_query: RunConfig traits:", file_group_config.to_dict()],
            indent=1,
            log_type="debug",
        )
        log_line(
            self.log,
            ["_internal_query: returning table rows", table.num_rows],
            indent=0,
            log_type="debug",
        )
        return file_group_config, table


    def get_files(self, config, output_path, file_name_pattern=None, source='parquet'):
        '''
        Retrieve files matching the group traits from the output directory.
        config: ProjectConfig object with project-level configurations
        output_path: Path object representing the output directory
        file_name_pattern: optional string pattern for file names
        source: string indicating the file format (default 'parquet')

        Populates:
            self.file_list: list of RunConfig objects for each file in the group
            self.internal_traits: dictionary of traits determined during file retrieval
            self.missing_files: dictionary of files that were expected but not found

        '''

        grp_path_template = config.get_dynamic_attr("output.{var}.dir_structure", source)  # config.output.grp_dir_structure
        if file_name_pattern is None:
            file_name_pattern = config.output.parquet.file_name#get_dynamic_attr("output.parquet.file_name{var}", "file_name_pattern")  # config.output.file_name_pattern

        grp_path_template_filled, replaced_parts = template_replace(grp_path_template, self.static_traits)
        log_line(self.log, ['DataGroup get_files: grp_path_template_filled:', grp_path_template_filled], indent=0, log_type="debug")

        # print('DataGroup get_files: grp_path_template_filled:', grp_path_template_filled, file=sys.stdout, flush=True)

        known_sections = grp_path_template_filled.split('/')
        bracket_locations = [ik for ik, section in enumerate(known_sections) if '{' in section]
        if len(bracket_locations) > 0:
            first_bracket_location = bracket_locations[0]
            _dir_known_section = '/'.join(known_sections[:first_bracket_location])
        else:
            _dir_known_section = '/'.join(known_sections)

        self.internal_traits = {key: value for key, value in self.static_traits.items() if (key not in replaced_parts)}
        for key in self.parent_config.traits:
            if key not in self.static_traits.keys():
                if key not in self.nonstatic_traits.keys():
                    self.internal_traits[key] = None

        merged_unaccounted_d = {**self.internal_traits, **self.nonstatic_traits}

        file_list = []
        missing_files = {}
        nonstatic_updates = defaultdict(set)
        for dirpath, _, filenames in os.walk(output_path / _dir_known_section):
            file_dir = Path(dirpath)
            if filenames:  # only keep dirs that contain files
                filtered_files = [file_dir/filename for filename in filenames if (f'.{source}' in filename) and
                                    ('registry' not in filename) and ('results.parquet' != filename) and ('.md' not in filename) and ('.yaml' not in filename) and (
                                          '.ipynb' not in filename) and ('.png' not in filename)]

                for file_path in filtered_files:
                    log_line(self.log, ['DataGroup get_files: checking file', file_path],
                             indent=0, log_type="debug")

                    # print('DataGroup get_files: checking file', file_path, file=sys.stdout, flush=True)
                    try:
                        file_traits = extract_from_pattern(file_path.name, file_name_pattern)
                        file_dict = {**{key: self.static_traits[key] for key in replaced_parts}, **file_traits}

                        fail = False

                        for trait_key in merged_unaccounted_d.keys():
                            if fail is False:
                                if trait_key in file_dict.keys():
                                    if merged_unaccounted_d[trait_key] is not None and file_dict[trait_key] not in correct_iterable(merged_unaccounted_d[trait_key]):
                                        fail = True
                                    else:
                                        nonstatic_updates[trait_key].add(file_dict[trait_key])

                        if fail is False:
                            new_config = self.parent_config.copy()
                            for key in self.nonstatic_traits.keys():
                                if (key not in file_dict.keys()) or (file_dict[key] is None):
                                    file_dict[key] = self.nonstatic_traits[key]
                            new_config.populate(file_dict)

                            try:
                                loaded_ds = ds.dataset(str(file_path), format="parquet")
                                log_line(self.log, ['get_files: loaded dataset for file', file_path],
                                         indent=0, log_type="debug")
                                # print('get_files: loaded dataset for file', file_path, file=sys.stdout, flush=True)
                                groupconfig_file, filtered_table  = self._internal_query(loaded_ds,
                                                                                    query_config=new_config)
                                log_line(self.log, ['get_files: filtered table rows after query', filtered_table.num_rows, 'for file', file_path,'fail status:', fail],
                                         indent=0, log_type="debug")
                                # print('get_files: filtered table rows after query', filtered_table.num_rows, 'for file', file_path,'fail status:', fail,file=sys.stdout, flush=True)
                            except:
                                filtered_table = None
                                fail=True

                            # print('fail status after filtering', fail, 'for file', file_path,'filtered_table', filtered_table, file=sys.stdout, flush=True)

                            if filtered_table is None:
                                # print('get_files: filtered table is None, failing for file', file_path, file=sys.stdout, flush=True)
                                log_line(self.log, ['get_files: filtered table is None, failing for file', file_path],
                                         indent=0, log_type="error")
                                fail = True

                            elif filtered_table.num_rows == 0:
                                log_line(self.log, ['get_files: filtered table has 0 rows, failing for file', file_path],
                                         indent=0, log_type="error")
                                # print('get_files: filtered table has 0 rows, failing for file', file_path, file=sys.stdout, flush=True)
                                fail = True

                        if fail is False:
                            groupconfig_file.output_path = [file_path]
                            # print('did not fail for file', file_path, file=sys.stdout, flush=True)
                            log_line(self.log, ['did not fail for file', file_path],
                                     indent=0, log_type="info")
                            file_list.append(groupconfig_file)

                            for key in groupconfig_file.traits:
                                new_values = correct_iterable(getattr(groupconfig_file, key)) if getattr(groupconfig_file, key) is not None else []
                                for val in new_values:
                                    nonstatic_updates[key].add(val)
                        else:
                            # missing_files.append((new_config, file_path))
                            if file_path not in missing_files.keys():
                                missing_files[file_path]=new_config
                            else:
                                for key in file_dict.keys():
                                    if key not in missing_files[file_path].keys():
                                        missing_files[file_path][key] = correct_iterable([key])
                                    else:
                                        missing_files[file_path][key] = list(
                                            set(correct_iterable(missing_files[file_path][key])) | set(
                                                correct_iterable(file_dict[key])))
                            # print('missing files', len(missing_files), file=sys.stdout, flush=True)
                            log_line(self.log, ['missing files', len(missing_files)],
                                     indent=0, log_type="info")

                    except ValueError as e:
                        log_line(self.log, e,
                                 indent=0, log_type="error")
                        # print(e, file=sys.stderr, flush=True)

        nonstatic_updates = {key: list(value) for key, value in nonstatic_updates.items()}
        for key in nonstatic_updates.keys():
            if len(nonstatic_updates[key]) == 1:
                self.static_traits[key] = nonstatic_updates[key][0]
            else:
                self.nonstatic_traits[key] = nonstatic_updates[key]
        self.file_list = file_list
        # print('DataGroup get_files: found', len(self.file_list), 'files', file=sys.stdout, flush=True)
        log_line(self.log, ['DataGroup get_files: found', len(self.file_list), 'files'],
                 indent=0, log_type="info")
        self.missing_files.update(missing_files)

    def pull_output(self, summary=True, full=False):
        '''
        Pull output data from the group files.
        '''
        tables = []
        print('pulling from datagrp')

        for ij, groupconfig_file in enumerate(self.file_list):
            filtered_table = groupconfig_file.pull_output(to_table=True)
            # print('pulled table rows', filtered_table.num_rows)
            log_line(self.log, ['pulled table rows', filtered_table.num_rows],
                     indent=0, log_type="info")
            if check_return(filtered_table) is True: tables.append(filtered_table)

        return OutputCollection(grp_specs=self.get_group_config(), in_table=tables, tmp_dir=self.tmp_dir) #if (len(tables) > 0) else None

    def get_metadata_as_iterables(self):
        self.metadata = {key: correct_iterable(value) for key, value in self.metadata.items()}

    def get_group_config(self):
        # return GroupConfig({**self.static_traits, **self.nonstatic_traits, **self.internal_traits})
        return RunConfig({**self.static_traits, **self.nonstatic_traits, **self.internal_traits})

#########################################

class Output:
    '''
    Output object to manage CCM output data.
    Methods:
        get_table(): load the output table from file if not already loaded
        clear_table(): release memory held by the output table and Arrow pools
        write_table(tag=''): write the output table to a Parquet file with an optional tag
    Attributes:
        _full: pyarrow Table object containing the full output data
        path: Path object representing the file path of the output data
        type: string indicating the type of output (e.g., 'delta_rho', 'libsize_aggregated')
        tmp_dir: Path object representing the temporary directory for intermediate files

    '''
    def __init__(self, full, path=None, outtype=None, tmp_dir=None):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if type(full) is pd.DataFrame:
            full = pa.Table.from_pandas(full, preserve_index=False)
        self._full = full
        self.path = path
        self.type = outtype
        self.tmp_dir = tmp_dir

    @property
    def surrogate(self):
        self.get_table()
        if 'surr_var' in self._full.schema.names:
            mask = pc.invert(pc.equal(self._full['surr_var'], 'neither'))
            surr_table = self._full.filter(mask)
            return surr_table
        else:
            return None

    @property
    def real(self):
        self.get_table()
        if 'surr_var' in self._full.schema.names:
            mask = pc.equal(self._full['surr_var'], 'neither')
            real_table = self._full.filter(mask)
            return real_table
        else:
            return self._full

    @property
    def table(self):
        self.get_table()
        return self._full

    @property
    def full(self):
        self.get_table()
        return self._full

    def get_table(self):
        if self._full is None:
            self._full = ds.dataset(str(self.path), format="parquet").to_table()

    def clear_table(self):
        """Release memory held by self.table and Arrow pools."""
        if self._full is not None:
            self._full = None
        gc.collect()
        pa.default_memory_pool().release_unused()

    def write_table(self, tag=''):
        if tag == '':
            tag = self.type if self.type is not None else 'scratch'
        if self.tmp_dir is None:
            self.tmp_dir = Path(os.getcwd()) / 'tmp'

        if self.path is None:
            unique_scratch_id = uuid.uuid4().hex
            unique_scratch_id = f'{unique_scratch_id}__{tag}'
            scratch_path = self.tmp_dir / f'{unique_scratch_id}.parquet'
            self.path = scratch_path

        if '__' not in str(self.path) and len(tag) >0:
            self.path = str(self.path).replace('.parquet', f'__{tag}.parquet')
        pq.write_table(self._full, self.path)


class OutputCollection:
    '''
    OutputCollection object to manage a collection of CCM output data.
    Methods:
        combine_OutputCollections(attr, other_output_collections): combine specified attribute from other OutputCollections
    Attributes:
        dyad_home: Path object representing the home directory for dyad analysis
        tmp_path: Path object representing the temporary directory for intermediate files
        grp_config: RunConfig object representing the group-level configuration
        label_stem: string representing the label stem for output files
        table: Output object containing the full output data
        libsize_aggregated: Output object containing libsize aggregated data
        active_stats: Output object containing active statistics data
        active_full: Output object containing active full data
        delta_rho_stats: Output object containing delta rho statistics data
        delta_rho_full: Output object containing delta rho full data
        relationships: Relationship object representing the relationships between variables
        r1: RelationshipSide object representing the first side of the relationship
        r2: RelationshipSide object representing the second side of the relationship

    '''
    def __init__(self, grp_specs=None, in_table=None, outtype=None, tmp_dir=None):
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

        self.relationships =  Relationship(self.grp_config.var_x, self.grp_config.var_y) if self.grp_config is not None else None
        self.r1 = RelationshipSide('r1', relationship=self.relationships) if self.relationships is not None else None
        self.r2 = RelationshipSide('r2', relationship=self.relationships) if self.relationships is not None else None

        if isinstance(grp_specs, RunConfig):
            self.grp_config = grp_specs
        elif isinstance(grp_specs, dict):
            iterable_d = {k: correct_iterable(v) for k, v in grp_specs.items()}
            self.grp_config = RunConfig(iterable_d)

        self.tmp_path = tmp_dir if tmp_dir is not None else (self.grp_config.proj_dir / 'tmp' if (self.grp_config is not None and self.grp_config.proj_dir is not None) else Path.cwd() / 'tmp')

        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.dyad_home = None
        # print('temporary directory for OutputCollection:', self.tmp_path)

            # iterable_d = {k: correct_iterable(v) for k, v in grp_specs.__dict__.items()}

    # def __init__(self, in_table):
        if isinstance(in_table, list) is False:
            if type(in_table) is pd.DataFrame:
                for col in ['E', 'tau', 'Tp', 'lag', 'knn', 'surr_var', 'surr_num', 'x_id', 'x_age_model_ind', 'x_var', 'y_id', 'y_age_model_ind', 'y_var', 'LibSize', 'ind_i', 'relation', 'forcing', 'responding']:
                    if col not in in_table.columns:
                        in_table[col] = self.grp_config.get_trait_value(col)
                in_table = pa.Table.from_pandas(in_table, preserve_index=False)
            in_table = [in_table]

        if isinstance(in_table, list) and (len(in_table)>0) and isinstance(in_table[0], pa.Table):
            tables = [tbl for tbl in in_table if (tbl is not None) and (isinstance(tbl, pa.Table) is True)]
            if len(tables) >0:
                in_table = pa.concat_tables(tables)
                self.table = Output(in_table, outtype=outtype, tmp_dir=self.tmp_path)
        elif isinstance(in_table, list) and (len(in_table)>0) and isinstance(in_table[0], Output):
            tables = [tbl.table for tbl in in_table if (tbl.table is not None) and isinstance(tbl.table, pa.Table) is True]
            if len(tables) >0:
                in_table = pa.concat_tables(tables)
                self.table = Output(in_table, outtype=outtype, tmp_dir=self.tmp_path)
        elif isinstance(in_table, list) and (len(in_table)>0) and isinstance(in_table[0], OutputCollection):
            outputcollections = [outputcoll for outputcoll in in_table if (outputcoll is not None) and (isinstance(outputcoll, OutputCollection) is True)]
            for attr in ['table', 'libsize_aggregated', 'active_stats', 'active_full', 'delta_rho_stats', 'delta_rho_full']:
                try:
                    self.combine_OutputCollections(attr, outputcollections)
                except Exception as e:
                    log_line(self.log, f'Error combining OutputCollections for attribute {attr}: {e}',
                             indent=0, log_type="error")
                    # print(f'Error combining OutputCollections for attribute {attr}: {e}')

        self.relationships = Relationship(self.grp_config.var_x, self.grp_config.var_y) if self.grp_config is not None else None

    def set_relationships(self):
        self.relationships = Relationship(self.grp_config.var_x, self.grp_config.var_y) if self.grp_config is not None else None
        self.r1 = RelationshipSide('r1', relationship=self.relationships) if self.relationships is not None else None
        self.r2 = RelationshipSide('r2', relationship=self.relationships) if self.relationships is not None else None

    def combine_OutputCollections(self, attr, other_output_collections):
        print('combining OutputCollections for', attr)
        tables = [getattr(self, attr)]
        print('combining', attr)
        if not isinstance(other_output_collections, list):
            other_output_collections = [other_output_collections]

        for other_output_collection in other_output_collections:
             tables.append(getattr(other_output_collection, attr))
        print(len(tables), 'tables to combine for', attr)
        tables = [tbl for tbl in tables if tbl is not None]
        col_types = {col: tables[0]._full.schema.field(col).type for col in tables[0]._full.schema.names}

        tables_full = []
        outtypes = []
        for tbl in tables:
            if (isinstance(tbl, Output) is True) and (tbl.table is not None) and (isinstance(tbl.table, pa.Table) is True):
                for col in tbl._full.schema.names:
                    if tbl._full.schema.field(col).type != col_types[col]:
                        tbl._full = tbl._full.set_column(
                            tbl._full.schema.get_field_index(col), col, tbl._full[col].cast(col_types[col]))
                tables_full.append(tbl.table)
                outtypes.append(tbl.type)
                tbl.clear_table()
            elif isinstance(tbl, pa.Table) is True:
                for col in tbl.schema.names:
                    if tbl.schema.field(col).type != col_types[col]:
                        tbl = tbl.set_column(
                            tbl.schema.get_field_index(col), col, tbl[col].cast(col_types[col]))
                tables_full.append(tbl)

        outtypes = list(set(outtypes))
        outtype = outtypes[0] if len(outtypes) == 1 else attr
        if len(tables_full)>0:
            setattr(self, attr, Output(pa.concat_tables(tables_full), outtype=outtype, tmp_dir=self.tmp_path))
            print('combined', attr)

        return self

    def calc_metrics(self, relationship_id=None, lag=None):
        self.delta_rho_stats.get_table()
        if relationship_id is None:
            try:
                self._calc_metrics('r1', lag=lag)
            except Exception as e:
                print(f'Error calculating metrics for r1: {e}')
            try:
                self._calc_metrics('r2', lag=lag)
            except Exception as e:
                print(f'Error calculating metrics for r2: {e}')
        else:
            self._calc_metrics(relationship_id=relationship_id, lag=lag)
        self.delta_rho_stats.clear_table()

    def _calc_metrics(self, relationship_id='r1', lag=None):
        gb_real = self.delta_rho_stats.real.group_by(["relation", 'lag', 'surr_var', 'surr_num']).aggregate(
            [("maxlibsize_rho", "mean"), ("delta_rho", "mean")])
        gb_real_df = gb_real.to_pandas()

        gb_surr = self.delta_rho_stats.surrogate.group_by(["relation", 'lag', 'surr_var', 'surr_num']).aggregate(
            [("maxlibsize_rho", "mean")])
        gb_surr_df = gb_surr.to_pandas()

        if relationship_id == 'r1':
            relationship = self.relationships.r1
        elif relationship_id == 'r2':
            relationship = self.relationships.r2

        real_r_df = gb_real_df[
            (gb_real_df['relation'] == relationship) & (gb_real_df['surr_var'] == 'neither')].reset_index(drop=True)


        if lag == 'pos':
            real_r_df = real_r_df[real_r_df['lag'] >= 0].reset_index(drop=True)
        elif lag == 'neg':
            real_r_df = real_r_df[real_r_df['lag'] <= 0].reset_index(drop=True)
        elif lag is not None:
            real_r_df = real_r_df[real_r_df['lag'] == lag].reset_index(drop=True)
        else:
            real_r_df = real_r_df

        real_r_ind = np.argmax(real_r_df['maxlibsize_rho_mean'].values)
        real_r_d = real_r_df.iloc[real_r_ind].to_dict()

        surr_rx_df = gb_surr_df[(gb_surr_df['relation'] == relationship) & (gb_surr_df['surr_var'] == self.relationships.var_x)]
        surr_rx_count = len(surr_rx_df.surr_num.unique())
        surr_rx_df_outperformers = surr_rx_df[surr_rx_df['maxlibsize_rho_mean'] > real_r_d['maxlibsize_rho_mean']]
        surr_rx_df_outperformers_count = len(surr_rx_df_outperformers.surr_num.unique())

        surr_ry_df = gb_surr_df[
            (gb_surr_df['relation'] == relationship) & (gb_surr_df['surr_var'] == self.relationships.var_y)]
        surr_ry_count = len(surr_ry_df.surr_num.unique())
        surr_ry_df_outperformers = surr_ry_df[surr_ry_df['maxlibsize_rho_mean'] > real_r_d['maxlibsize_rho_mean']]
        surr_ry_df_outperformers_count = len(surr_ry_df_outperformers.surr_num.unique())

        if relationship_id == 'r1':
            self.r1.surr_rx_count = surr_rx_count
            self.r1.surr_rx_count_outperforming = surr_rx_df_outperformers_count
            self.r1.surr_ry_count = surr_ry_count
            self.r1.surr_ry_count_outperforming = surr_ry_df_outperformers_count
            self.r1.delta_rho = real_r_d['delta_rho_mean']
            self.r1.maxlibsize_rho = real_r_d['maxlibsize_rho_mean']
            self.r1.lag = real_r_d['lag']
            self.r1.surr_rx_outperforming_frac = surr_rx_df_outperformers_count / surr_rx_count if surr_rx_count > 0 else None
            self.r1.surr_ry_outperforming_frac = surr_ry_df_outperformers_count / surr_ry_count if surr_ry_count > 0 else None
        elif relationship_id == 'r2':
            self.r2.surr_rx_count = surr_rx_count
            self.r2.surr_rx_count_outperforming = surr_rx_df_outperformers_count
            self.r2.surr_ry_count = surr_ry_count
            self.r2.surr_ry_count_outperforming = surr_ry_df_outperformers_count
            self.r2.delta_rho = real_r_d['delta_rho_mean']
            self.r2.maxlibsize_rho = real_r_d['maxlibsize_rho_mean']
            self.r2.lag = real_r_d['lag']
            self.r2.surr_rx_outperforming_frac = surr_rx_df_outperformers_count / surr_rx_count if surr_rx_count > 0 else None
            self.r2.surr_ry_outperforming_frac = surr_ry_df_outperformers_count / surr_ry_count if surr_ry_count > 0 else None

    def calc_delta_rho(self, *, stats_out=True, full_out=False, **kwargs):
        """
        Iterates unique combinations of calc_grp_cols and applies compute_delta_rho_arrow
        to each group's sub-table. Returns concatenated Arrow tables.
        """
        # Get unique groups as a small table
        full = self.table.full

        group_traits_below = self.grp_config.trait_hierarchy(full, 'LibSize', level="below", threshold=0.8, include_ids=True)
        # print('group_traits_below', group_traits_below)

        calc_grp_cols = [col for col in full.schema.names if col in self.grp_config.traits and (
                         col not in group_traits_below)]  # if self.grp_config.traits if (col in self.full.schema.names) and (col not in "output_path")]

        if 'relation' in full.schema.names:
            if 'relation' not in calc_grp_cols:
                calc_grp_cols.append('relation')

        unique_tbl = full.select(calc_grp_cols).combine_chunks().group_by(calc_grp_cols).aggregate([(calc_grp_cols[0], "count")]).select(calc_grp_cols)

        stats_tables = []
        full_tables = []
        for row_idx in range(unique_tbl.num_rows):
            # try:
            gd = {}
            for col in calc_grp_cols:
                val = unique_tbl[col][row_idx]
                vals = correct_iterable(val.as_py())
                gd[col] = vals
            filter_fail = False
            try:
                filters = [pc.field(col).isin(correct_iterable(unique_tbl[col][row_idx].as_py())) for col in calc_grp_cols]
                combined_filter = reduce(operator.and_, filters)
                grp_tbl = full.filter(combined_filter)
                filter_fail = False
            except Exception as e:
                print(gd, e)
                filter_fail = True

            if filter_fail is True:
                continue

            s_tbl, f_tbl = compute_delta_rho_grp(
                grp_tbl, gd, stats=stats_out, full=full_out, **kwargs
            )
            if stats_out is True and s_tbl is not None and s_tbl.num_rows > 0:
                stats_tables.append(s_tbl)
            if full_out is True and f_tbl is not None and f_tbl.num_rows > 0:
                full_tables.append(f_tbl)

        if stats_out is True:
            out_stats = pa.concat_tables(stats_tables) if stats_tables else None
            self.delta_rho_stats = Output(out_stats, outtype='delta_rho_stats', tmp_dir=self.tmp_path)#, use_case='delta_rho_stats')
        if full_out is True:
            out_full = pa.concat_tables(full_tables) if full_tables else None
            self.delta_rho_full = Output(out_full, outtype='delta_rho_full', tmp_dir=self.tmp_path)#, use_case='delta_rho_full')

        return self

    def aggregate_libsize(self, query_config=None): #process_group_table
        knn = get_static(query_config.knn if query_config is not None else self.grp_config.knn)
        full = self.table.full
        if isinstance(self.table.full, pd.DataFrame):
            full = pa.Table.from_pandas(self.table.full)

        if "LibSize" in full.schema.names:
            mask = pc.greater(full["LibSize"], knn+1)
            group_table = full.filter(mask)
        if group_table.num_rows == 0:
            return self

        calc_grp_cols = ['E', 'tau', 'Tp', 'lag', 'knn', 'surr_var', 'surr_num', 'x_id', 'x_age_model_ind', 'x_var', 'y_id', 'y_age_model_ind', 'y_var', 'LibSize', 'ind_i', 'relation', 'forcing', 'responding']

        if "LibSize" in full.schema.names:
            if "LibSize" not in calc_grp_cols:
                calc_grp_cols.append("LibSize")
        if 'relation' in full.schema.names:
            if 'relation' not in calc_grp_cols:
                calc_grp_cols.append('relation')

        aggregated_cols = [col for col in full.schema.names if (col not in calc_grp_cols) and ('id' not in col) and ('ind' not in col) and (full[col].type in [pa.float32(), pa.float64(), pa.int32(), pa.int64()])]
        # print('aggregated cols', aggregated_cols)
        log_line(self.log, ['aggregated cols', aggregated_cols],
                 indent=0, log_type="debug")
        grouped_aggregated_table = pa.TableGroupBy(full, calc_grp_cols).aggregate([(col, "mean") for col in aggregated_cols])
        new_names = [col.replace('_mean', '') for col in grouped_aggregated_table.schema.names]
        grouped_aggregated_table = grouped_aggregated_table.rename_columns(new_names)
        self.libsize_aggregated = Output(grouped_aggregated_table, outtype='libsize_aggregated', tmp_dir=self.tmp_path)#, use_case='libsize_aggregated')
        return self

    def clear_tables(self):
        """Release memory held by all tables and Arrow pools."""
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
        pa.default_memory_pool().release_unused()

    def get_table_paths(self):
        '''
        Retrieve file paths for all stored tables in the OutputCollection.
        '''
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

    def migrate_path(self, new_dyad_home=None, tmp_home=None):
        '''
        Migrate all stored table paths to a new dyad home and temporary directory.
        Because the OutputCollection operates by reading in and clearing tables, paths must be updated when the dyad home or temporary directory changes.

        Assumptions:
        - new_dyad_home is the path to parent of the dyad directory (so the directory housing, for example, Erb22daGMST_Wu18TSI)
        - tmp_home is the name of the dyad directory (e.g. Erb22daGMST_Wu18TSI)

        '''
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



#
# class DataVarConfig:
#     def __init__(self, config, var_id, proj_dir, suffix_label=None, suffix_ind=None):
#
#         self.var_id = var_id
#         self.suffix_label= suffix_label if suffix_label is not None else ''
#         self.suffix_ind = suffix_ind if suffix_ind is not None else ''
#         self.suffix = f'{self.suffix_label}{self.suffix_ind}'
#
#         self.raw_data_csv = None
#         self.raw_data_var = None
#         self.raw_data_col = None
#         self.raw_time_var = None
#         self.var = None  # e.g. 'temp'
#
#         self.surr_csvs = None
#         self.surr_csv_stem = None
#         self.surr_csv = None
#         self.surr_time_var = None
#         self.surr_prefix = None
#         self.surr_var = None  # e.g. 'temp'
#         # self.surr_num = None
#
#         self.obs_type = None
#         self.source = None
#         self.unit = None
#         self.var_label = None
#         self.var_name = None
#         self.year = None
#         self.color = None
#
#         # TODO there is some future redundancy here and sketchy path handling
#         self.raw_data_dir_path = None
#         self.surr_data_dir_path = None
#         self.proj_dir = None
#         self.populate(config, proj_dir)
#
#
#     def populate(self, config, proj_dir):
#
#         self.proj_dir = proj_dir
#         try:
#             var_yaml = config.get_dynamic_attr("data_vars.{var}.config", self.var_id)
#             # load variable-specific settings from config
#             self.load_from_var_yaml(var_yaml, proj_dir)
#             var_info = var_yaml.get(self.var_id, None) if var_yaml is not None else None
#         except:
#             print(f'reading var yaml for {self.var_id} failed, trying config')
#             self.load_from_config( config, proj_dir)
#
#     def load_from_var_yaml(self, var_yaml, proj_dir):
#         print('load_from_var_yaml function is a stub - needs to be implemented')
#         pass
#
#     # TODO fix pointers for surrogates
#     def load_from_config(self, config, proj_dir):
#
#         var_id = self.var_id
#         var_info = config.get_dynamic_attr("{var}", self.var_id)
#         var_info = var_info.to_dict()
#
#         real_ts_d = var_info.pop('real_ts', None)
#         surr_ts_d = var_info.pop('surrogate_ts', None)
#
#
#         if 'raw_data_var' not in var_info.keys():
#             if 'data_var' in var_info.keys():
#                 data_var = var_info.pop('data_var', None)
#                 var_info['raw_data_var'] = data_var
#
#         if 'raw_data_csv' not in var_info.keys():
#             if 'data_csv' in var_info.keys():
#                 data_csv = var_info.pop('data_csv', None)
#                 var_info['raw_data_csv'] = data_csv
#
#         if 'raw_time_var' not in var_info.keys():
#             time_var = var_info.pop('raw_time_var', None)
#             if 'time_var' in var_info.keys():
#                 time_var = var_info.pop('time_var', None)
#             else:
#                 time_var = 'time'
#             var_info['raw_time_var'] = time_var
#
#         if 'surr_time_var' not in var_info.keys():
#             var_info['surr_time_var']='date'
#
#
#         if 'surr_var' not in var_info.keys() or var_info['surr_var'] is None:
#             var_info['surr_var'] = var_info.get('var', None)
#
#         try:
#             surr_csvs = config.get_dynamic_attr("{var}.surr_file_name", self.var_id)
#         except:
#             surr_csvs = None
#
#         if surr_csvs is not None:
#             surr_csvs = correct_iterable(surr_csvs)
#             if len(surr_csvs) == 1:
#                 var_info['surr_csv_stem'] = surr_csvs[0].replace('.csv', '').replace('.txt', '')
#             else:
#                 print(f'Multiple surrogate csvs found for {self.var_id}: {surr_csvs}')
#
#         var_info['surr_prefix'] = var_info.get('surr_prefix', var_info.get('surr_var', None))
#         for key in var_info.keys():
#             if hasattr(self, key):
#                 setattr(self, key, var_info[key])
#
#         self.raw_data_dir_path = self.set_data_source(config, data_source='data', data_type='raw')
#         self.get_color(config)
#         self.set_surr_csv_name()
#         self.surr_data_dir_path = self.set_data_source(config, data_source='data', data_type='surr')
#         self.set_raw_data_col()
#
#     def set_surr_csv_name(self):
#         if len(self.suffix) >0:
#             self.surr_csv = '__'.join([self.surr_csv_stem, self.suffix]).strip('__') if self.surr_csv_stem is not None else None
#         else:
#             self.surr_csv = self.surr_csv_stem
#
#     def set_raw_data_col(self):
#         if len(self.suffix) > 0:
#             self.raw_data_col = '__'.join([self.raw_data_var, self.suffix]).strip('__') if self.raw_data_var is not None else None
#         else:
#             self.raw_data_col = self.raw_data_var
#
#     def set_data_source(self, config,data_source='data' , var_data_csv=None, data_type='raw'):
#         if var_data_csv is None:
#             if data_type == 'raw':
#                 var_data_csv = self.raw_data_csv
#             elif data_type in ['surr', 'surrogate']:
#                 var_data_csv = self.surr_csv
#
#         data_path, _ = choose_data_source(self.proj_dir, config, data_source, data_type=data_type, var_data_csv=var_data_csv)
#         data_path = Path(data_path).parent
#         return data_path
#
#     def get_color(self, config):
#         if self.color is None:
#             color_map = config.pal.to_dict()
#             if color_map is not None and self.var_id in color_map:
#                 self.color = color_map[self.var_id]
#             else:
#                 self.color = 'black'
#
#
# class VarObject(DataVarConfig):
#     def __init__(self, config, var_id=None, proj_dir=None, data_var_config=None):
#         if data_var_config is not None and isinstance(data_var_config, DataVarConfig):
#             # Copy all attributes from the provided DataVarConfig
#             for key, value in data_var_config.__dict__.items():
#                 setattr(self, key, value)
#         else:
#             # Initialize as a new DataVarConfig
#             super().__init__(config, var_id, proj_dir)
#
#         self.ts = None
#         self.ts_type = None # 'real' or 'surr'
#         self.surr_num = None
#         self.col_name = None
#         self.time_var = None
#
#     def set_col_name(self):
#         if self.ts_type == 'raw':
#             self.col_name = self.raw_data_col
#         elif self.ts_type == 'surr':
#             self.col_name = f'{self.surr_prefix}_{self.surr_num}'
#
#     def standardize_time_var(self, specified_time_var, df, other_col):
#
#         if ('time' not in df.columns) and (specified_time_var is not None):
#             df = df.rename(columns={specified_time_var: 'time'})
#         if 'date' in df.columns:
#             df = df.rename(columns={'date': 'time'})
#         df['time'] = df['time'].astype('int')
#
#         return df, 'time'
#
#     def get_raw(self):
#         # get raw timeseries data from csv
#         self.ts_type = 'raw'
#         self.set_col_name()
#
#         if (self.raw_data_dir_path/check_csv(self.raw_data_csv)).exists() is True:
#             raw_data = pd.read_csv(self.raw_data_dir_path/check_csv(self.raw_data_csv))
#             # print('raw data read', raw_data.head())
#             raw_data = remove_extra_index(raw_data)
#             # print('raw data before standardize', raw_data.head())
#
#             raw_data, time_var = self.standardize_time_var(self.raw_time_var, raw_data, self.col_name)
#             self.time_var = time_var
#             # print('raw data', raw_data.head())
#
#             self.ts = raw_data[[self.time_var, self.col_name]].copy()
#
#     def get_surr(self, surr_num=None):
#         # print('sur', self.surr_data_dir_path / check_csv(self.surr_csv))
#         if (self.surr_data_dir_path / check_csv(self.surr_csv)).exists() is True:
#             surr_data = pd.read_csv(self.surr_data_dir_path / check_csv(self.surr_csv))
#             surr_data = remove_extra_index(surr_data)
#             # print(surr_data)
#
#             # self.surr_num = self.surr_num if self.surr_num is not None else surr_num
#             self.set_col_name()
#             self.ts_type = 'surr'
#
#             surr_data, time_var = self.standardize_time_var(self.raw_time_var, surr_data, self.col_name)
#             self.time_var = time_var
#             # print('surr data', surr_data[[self.time_var, self.col_name]].head())
#             self.ts = surr_data[[self.time_var, self.col_name]].copy()
#             # print(self.ts.head())

def merge_variable_ts(col_var_obj, target_var_obj):
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

    def __init__(self, grp_specs, config, proj_dir=None, tmp_dir=None, exclusion_radius=None, init_var_objs=True):
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
        self.df = merge_variable_ts(self.col_var_obj, self.target_var_obj)
        self.df = self.df.iloc[self.train_ind_i : self.train_ind_f].reset_index(drop=True) if self.train_ind_f is not None else self.df.iloc[self.train_ind_i : ].reset_index(drop=True)
        return self


class CCMConfig(CMConfigBase):

    def __init__(self, grp_specs, config, proj_dir=None, cpus=1, exclusion_radius=None, limit_surr_libsizes= True):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        rc = RunConfig(grp_specs)
        super().__init__(grp_specs, config, proj_dir=proj_dir, exclusion_radius=exclusion_radius, init_var_objs=True)

        self.file_name = self.get_filename(config)

        self.df = None
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
        # generate filename of CCM CSV based on template in config
        pset_d = self.to_dict()
        try:
            file_name_template = config.output.csv.file_format
            file_name = template_replace(file_name_template, pset_d, return_replaced=False)# f'{replace(file_name_template, pset_d)}.csv'
        except:
            file_name = f"{pset_d['pset_id']}_E{pset_d['E']}_tau{pset_d['tau']}__{pset_d['surr_var']}{pset_d['surr_num']}.csv"

        return check_csv(file_name)

    def check_run_exists(self):

        pset_exists, stem_exists = check_exists(check_csv(self.file_name), Path(self.output_path))
        if self.output_path is None or self.file_name is None:
            return False

        if pset_exists != self.file_path.exists():
            print(f'Warning: mismatch between expected existence {pset_exists} and actual existence {self.file_path.exists()} for {self.file_path}')

        print(f'Checking existence of CCM output at {self.file_path}: {pset_exists}')
        return pset_exists, stem_exists

    def set_output_calc_sub(self, config, output_dir, file_name):

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
        if self.min_window is not None:
            self.libsizes = np.concatenate([np.arange(self.min_libsize, self.min_libsize+self.min_window, self.libsize_step),
                                            np.arange(self.max_libsize -self.min_window, self.max_libsize, self.libsize_step)])
        else:
            self.libsizes = np.arange(self.min_libsize, self.max_libsize + 1, self.libsize_step)

    def set_col_ts(self, surr_num=None):
        if self.col_var_obj.ts_type == 'surr':
            if (self.col_var_obj.surr_num is None) and (surr_num is not None):
                self.col_var_obj.surr_num = surr_num

        if self.col_var_obj.surr_num not in (0, None):
            self.col_var_obj.get_surr(self.col_var_obj.surr_num)
        else:
            self.col_var_obj.get_real()

    def set_target_ts(self, surr_num=None):
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

    # def make_time_embedding
    # def make_depth_embedding

    def shift(self):
        shifted = self.df.copy()
        shifted[self.target_var] = shifted[self.target_var].shift(self.lag)
        shifted = shifted.dropna()

        self.train_ind_f = shifted.index.values[-1] #if self.train_ind_f is None else self.train_ind_f

        self.df = shifted.reset_index(drop=True)
        self.max_libsize = min(self.max_libsize, int(.75*len(self.df)))

    def run_ccm(self, overwrite=None, ind=None, args=None, script=None):

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
#
# class RelationshipSide:
#     def __init__(self, r, relationship=None, var_x='temp', var_y='TSI', influence_word='causes'):
#         self.var_x = var_x if relationship is None else relationship.var_x
#         self.var_y = var_y if relationship is None else relationship.var_y
#         self.influence_word = influence_word
#
#         self.surr_rx_count = None
#         self.surr_rx_count_outperforming = None
#         self.surr_ry_count = None
#         self.surr_ry_count_outperforming = None
#         self.delta_rho = None
#         self.maxlibsize_rho = None
#         self.lag = None
#         self.surr_rx_outperforming_frac = None
#         self.surr_ry_outperforming_frac = None
#
#
#         # self.surr_rx
#         # self.surr_ry
#
#         if r == 'r1':
#             self.pattern = 'y causes x'
#         elif r == 'r2':
#             self.pattern = 'x causes y'
#
#
#     @property
#     def surr_rx(self):
#         return self.pattern.replace('x', f'{self.var_x} (surr)').replace('y', self.var_y).replace('causes', self.influence_word)
#
#     @property
#     def surr_ry(self):
#         return self.pattern.replace('y', f'{self.var_y} (surr)').replace('x', self.var_x).replace('causes', self.influence_word)
#
#     @property
#     def r(self):
#         return self.pattern.replace('x', self.var_x).replace('y', self.var_y).replace('causes', self.influence_word)
#
#
#
# class Relationship:
#
#     def __init__(self, var_x='temp', var_y='TSI', surr_flag='neither'):
#
#         self.influence_word = 'causes'
#         self.var_x = var_x
#         self.var_y = var_y
#         self.surr_flag = surr_flag
#
#         # self.active_r1 = self.set_active_r1()
#         # self.active_r2 = self.set_active_r2()
#
#
#     def set_influence_verb(self, verb):
#         self.influence_word = verb
#
#
#     def set_active_r1(self):
#         if self.surr_flag in ('x', self.var_x):
#             return self.surr_r1x
#         elif self.surr_flag in ('neither'):
#             return self.r1
#         elif self.surr_flag in ('y', self.var_y):
#             return self.surr_r1y
#         elif self.surr_flag in ('both'):
#             return self.surr_r1yx
#
#
#     def set_active_r2(self):
#         if self.surr_flag in ('x', self.var_x):
#             return self.surr_r2x
#         elif self.surr_flag in ('neither'):
#             return self.r2
#         elif self.surr_flag in ('y', self.var_y):
#             return self.surr_r2y
#         elif self.surr_flag in ('both'):
#             return self.surr_r2yx
#
#     @property
#     def r1(self):
#         return f'{self.var_y} {self.influence_word} {self.var_x}'
#
#     @property
#     def r2(self):
#         return f'{self.var_x} {self.influence_word} {self.var_y}'
#
#     @property
#     def surr_r1x(self):
#         return f'{self.var_y} {self.influence_word} {self.var_x} (surr)'
#
#     @property
#     def surr_r1y(self):
#         return f'{self.var_y} (surr) {self.influence_word} {self.var_x}'
#
#     @property
#     def surr_r2x(self):
#         return f'{self.var_x} (surr) {self.influence_word} {self.var_y}'
#
#     @property
#     def surr_r2y(self):
#         return f'{self.var_x} {self.influence_word} {self.var_y} (surr)'
#
#     @property
#     def surr_r2xy(self):
#         return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'
#
#     @property
#     def surr_r2yx(self):
#         return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'
#
#     @property
#     def surr_r2both(self):
#         return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'
#
#     @property
#     def surr_r1xy(self):
#         return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'
#
#     @property
#     def surr_r1yx(self):
#         return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'
#
#     @property
#     def surr_r1both(self):
#         return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'
#
#
#
