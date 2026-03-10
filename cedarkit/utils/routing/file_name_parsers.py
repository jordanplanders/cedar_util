import re
import logging
logger = logging.getLogger(__name__)
# from .location_helpers import check_location
try:
    from cedarkit.utils.routing.paths import check_location
    from cedarkit.utils.cli import setup_logging, log_line
except ImportError:
    # Fallback: imports when running as a package
    from utils.routing.paths import check_location
    from utils.cli.logging import setup_logging, log_line


# String methods for file names and parameter values
def check_csv(output_file_name):
    if '.csv' not in output_file_name:
        output_file_name = f'{output_file_name}.csv'
    return output_file_name

def remove_numbers(input_string):
    # Use regex to remove all digits from the string
    return re.sub(r'\d+', '', input_string)

def extract_numbers(input_string):
    # Use regex to extract all numbers from the string
    return re.findall(r'\d+', input_string)[-1]  # Assuming numbers are before the first '__'


# # TODO migrate to DataVar
# def pull_raw_data(config, proj_dir, var_ids, alias=True, data_source='data'):
#     """
#     Pulls raw data for specified variable IDs from the configuration and merges them into a single DataFrame.
#     Args:
#         config (Config): The configuration object containing data source information.
#         proj_dir (Path): The project directory path.
#         var_ids (List[str]): List of variable IDs to pull data for.
#         alias (bool): Whether to use variable aliases from the config. Default is True.
#         data_source (str): The data source to use ('data', 'raw_data', 'master_data', 'master'). Default is 'data'.
#     Returns:
#         pd.DataFrame: Merged DataFrame containing time variable and specified variables.
#
#     Notes:
#         1. The function retrieves the time variable name and its alias from the config.
#         2. For each variable ID, it fetches the corresponding data variable name and its alias (if alias=True).
#         3. It attempts to read the data from the specified data source, handling potential errors.
#         4. The function checks for the presence of the time variable in the data and adjusts if necessary.
#         5. It merges all individual variable DataFrames on the time variable, dropping rows with all NaN values in the specified variables.
#         6. Returns the final merged DataFrame.
#
#     Calls:
#         choose_data_source(proj_dir, config, data_source, var_data_csv)
#         check_time_var(var_data, time_var)
#     """
#
#     time_var = config.raw_data.time_var
#     time_var_alias = 'date'#config.raw_data.time_var_alias
#
#     data_dfs = []
#     var_aliases = []
#     for var_id in var_ids:
#         try:
#             data_var = config.get_dynamic_attr("{var}.real_ts_var", var_id)
#         except:
#             data_var = config.get_dynamic_attr("{var}.data_var", var_id)
#         if alias == True:
#             data_var_alias = config.get_dynamic_attr("{var}.var", var_id)
#         else:
#             data_var_alias = data_var
#         var_aliases.append(data_var_alias)
#
#         try:
#             var_data_csv = config.get_dynamic_attr("{var}.raw_data_csv", var_id)
#         except:
#             var_data_csv = config.get_dynamic_attr("{var}.data_csv", var_id)
#         data_path, var_data = choose_data_source(proj_dir, config, data_source, var_data_csv=var_data_csv)
#         print(f'Using data from {data_path}', file=sys.stdout, flush=True)
#
#         if var_data is None:
#             print(f'Error reading raw data for {var_id} from {data_path}', file=sys.stderr, flush=True)
#             continue
#
#         if time_var not in var_data.columns:
#             time_var = check_time_var(var_data, time_var)
#
#         var_data = var_data[[time_var, data_var]].rename(columns=
#                                                                          {time_var: time_var_alias,
#                                                                           data_var: data_var_alias,
#                                                                           })
#         var_data = var_data.dropna(subset=[data_var_alias], how='all')
#         var_data = var_data[[time_var_alias, data_var_alias]].copy()
#         data_dfs.append(var_data)
#
#     data = data_dfs[0]
#     for var_df in data_dfs[1:]:
#         data = pd.merge(data, var_df, on=time_var_alias, how='outer')
#
#     data = data.dropna(subset=var_aliases, how='all')
#     # print(f'Raw data shape: {data.shape}, {data.head}', file=sys.stdout, flush=True)
#     return data


def parse_surr_label(token: str, x_var: str, y_var: str):
    """
    Parses suffix token to determine surrogate variable and index.
    Parameters:
        token (str): Suffix token indicating surrogate type and index. (e.g. 'neither0', 'TSI147', 'temp033', 'both12').
        x_var (str): Name of the first variable (e.g., 'temp').
        y_var (str): Name of the second variable (e.g., 'TSI').

    Returns:
        (surr_var, surr_num) where surr_var is inferred from the token.
        Recognizes 'neither', 'both', x_var/y_var specific labels, and generic
        labels like '<var><digits>' (e.g., 'temp17').

    Used by:
        package_calc_grp_results_to_parquet
    """

    lab_l = token.lower()
    xv, yv = x_var.lower(), y_var.lower()

    if "neither" in lab_l:
        return "neither", 0
    if "both" in lab_l:
        return "both", 99999 # use a large number to indicate both but without a specific index

    for needle, out_var in ((xv, x_var), (yv, y_var)):
        m = re.fullmatch(rf'\s*({re.escape(needle)})(\d+)\s*', lab_l)
        if m:
            return out_var, int(m.group(2))

    # Generic fallback for labels that don't match x_var/y_var exactly, e.g. temp17.
    # This keeps packaging robust for legacy/misaligned naming.
    m_generic = re.fullmatch(r"\s*([A-Za-z][\w-]*)(\d+)\s*", token)
    if m_generic:
        return m_generic.group(1), int(m_generic.group(2))

    # if xv in lab_l:
    #     num = lab_l.replace(xv, '').strip()
    #     if num.isdigit():
    #         num = int(num)
    #     return x_var, num
    # if yv in lab_l:
    #     num = lab_l.replace(yv, '').strip()
    #     if num.isdigit():
    #         num = int(num)
    #     return y_var, num
    print(f"Warning: Unrecognized suffix token '{token}', skipping")
    return None


def extract_from_pattern(filename: str, pattern_str: str):
    """
    Extracts parameter values from filename based on a pattern string.
    Example:
        extract_from_pattern("E4_tau1_lag-5.parquet", "E{E}_tau{tau}_lag{lag}")
        -> {'E': 4, 'tau': 1, 'lag': -5}
    """
    # Convert format specifiers like {E}, {tau}, {lag} into named regex groups
    # regex = re.sub(r"\{(\w+)\}", lambda m: f"(?P<{m.group(1)}>-?\\d+)", pattern_str)
    regex = re.sub(
        r"\{(\w+)\}",
        lambda m: f"(?P<{m.group(1)}>[-+]?\d+(?:\.\d+)?|[A-Za-z_][\\w-]*)",
        pattern_str
    )

    match = re.search(regex, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match pattern '{pattern_str}'")

    # Convert all extracted values to integers
    parts_d = {k: v for k, v in match.groupdict().items()}
    parts_d = {k: int(v) if v.lstrip('-').isdigit() else v for k, v in parts_d.items()}
    return parts_d

#
def template_replace(template, d, return_replaced=True):
    replaced = []
    old_template = template
    for key, value in d.items():
        template = template.replace(f'{{{key}}}', str(value))
        if template != old_template:
            replaced.append(key)
            old_template = template
    if return_replaced is False:
        return template

    return template, replaced
