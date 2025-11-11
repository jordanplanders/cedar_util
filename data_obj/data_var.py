from pathlib import Path
import pandas as pd
import numpy as np
import pyleoclim as pyleo

from utils.data_access import choose_data_source, check_csv, remove_extra_index
from utils.config_parser import load_config


class DataVarConfig:
    def __init__(self, config, var_id, proj_dir, suffix_label=None, suffix_ind=None):

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
        self.var_label = None
        self.var_name = None
        self.year = None
        self.color = None

        # TODO there is some future redundancy here and sketchy path handling
        self.real_data_dir_path = None
        self.surr_data_dir_path = None
        self.proj_dir = None

        # self.real_ts_col = None
        self.real_ts_csv = None
        self.surr_ts_csv = None

        self.populate(config, proj_dir)

    def populate(self, config, proj_dir):
        self.proj_dir = proj_dir
        try:
            var_yaml = config.get_dynamic_attr("data_vars.{var}", self.var_id)
            var_info = load_config(proj_dir / 'var_configs'/f'{var_yaml}.yaml')

            # load variable-specific settings from config
            # self.load_from_var_yaml(var_yaml, proj_dir)
            var_info = var_yaml.get(self.var_id, None) if var_yaml is not None else None
        except:
            print(f'reading var yaml for {self.var_id} failed, trying config')
            var_info = config.get_dynamic_attr("{var}", self.var_id)
            var_info = var_info.to_dict()
            # self.load_from_config(config, proj_dir)

        real_ts_d = var_info.pop('real_data_ts', None)
        # real_csv_stem
        if 'real_csv_stem' in var_info.keys():
            self.real_csv_stem = var_info.pop('real_csv_stem', None)
        elif 'csv_stem' in real_ts_d.keys():
            self.real_csv_stem = real_ts_d.pop('csv_stem', None)
        elif 'data_csv' in var_info.keys():
            self.real_csv_stem = var_info.pop('data_csv', None)
        else:
            print(f'No real_csv_stem found for {self.var_id}')

        # real_ts_var
        if 'real_ts_var' in var_info.keys():
            self.real_ts_var = var_info.pop('real_ts_var', None)
        elif 'var' in real_ts_d.keys():
            self.real_ts_var = real_ts_d.pop('var', None)
        elif 'data_var' in var_info.keys():
            self.real_ts_var = var_info.pop('data_var', None)
        else:
            print(f'No real_data_var found for {self.var_id}')

        # real_ts_time
        if 'real_ts_time' in var_info.keys():
            self.real_ts_time = var_info.pop('real_time_var', None)
        elif 'time' in real_ts_d.keys():
            self.real_ts_time = real_ts_d.pop('time', None)
        elif 'time_var' in var_info.keys():
            self.real_ts_time = var_info.pop('time_var', None)
        else:
            self.real_ts_time = 'time'

        self.set_real_csv_name()
        self.real_data_dir_path = self.set_data_source(config, data_source='data', data_type='real')
        self.get_color(config)

        surr_ts_d = var_info.pop('surrogate_ts', None)
        #surr_csv_stem
        if 'surr_csv_stem' in var_info.keys():
            self.surr_csv_stem = var_info.pop('surr_csv_stem', None)
        elif 'csv_stem' in surr_ts_d.keys():
            self.surr_csv_stem = surr_ts_d.pop('csv_stem', None)
        elif 'surr_file_name' in var_info.keys():
            surr_file_name = var_info.pop('surr_file_name', None)
            self.surr_csv_stem = surr_file_name.replace('.csv', '').replace('.txt', '')
        else:
            print(f'No surr_csv_stem found for {self.var_id}')

        # surr_ts_var
        if 'surr_var' in var_info.keys():
            self.surr_ts_var = var_info.pop('surr_var', None)
        elif 'var' in surr_ts_d.keys():
            self.surr_ts_var = surr_ts_d.pop('var', None)
        else:
            self.surr_ts_var = self.var

        # surr_ts_time
        if 'surr_time_var' in var_info.keys():
            self.surr_ts_time = var_info.pop('surr_time_var', None)
        elif 'time' in surr_ts_d.keys():
            self.surr_ts_time = surr_ts_d.pop('time', None)
        else:
            self.surr_ts_time = 'date'

        self.set_surr_csv_name()
        self.surr_data_dir_path = self.set_data_source(config, data_source='data', data_type='surr')

        for key in var_info.keys():
            if hasattr(self, key):
                if getattr(self, key) is None:
                    setattr(self, key, var_info[key])

    def set_surr_csv_name(self):
        if len(self.suffix) > 0:
            self.surr_ts_csv = '__'.join([self.surr_csv_stem, self.suffix]).strip(
                '__') if self.surr_csv_stem is not None else None
        else:
            self.surr_ts_csv = self.surr_csv_stem

    def set_real_csv_name(self):
        self.real_ts_csv = self.real_csv_stem

    def set_data_source(self, config, data_source='data', var_data_csv=None, data_type='real'):
        if var_data_csv is None:
            if data_type == 'real':
                var_data_csv = self.real_ts_csv
            elif data_type in ['surr', 'surrogate']:
                var_data_csv = self.surr_ts_csv

        data_path, _ = choose_data_source(self.proj_dir, config, data_source, data_type=data_type,
                                          var_data_csv=var_data_csv)
        data_path = Path(data_path).parent
        return data_path

    def get_color(self, config):
        if self.color is None:
            color_map = config.pal.to_dict()
            if color_map is not None and self.var_id in color_map:
                self.color = color_map[self.var_id]
            else:
                self.color = 'black'


class VarObject(DataVarConfig):
    def __init__(self, config, var_id=None, proj_dir=None, data_var_config=None):
        if data_var_config is not None and isinstance(data_var_config, DataVarConfig):
            # Copy all attributes from the provided DataVarConfig
            for key, value in data_var_config.__dict__.items():
                setattr(self, key, value)
        else:
            # Initialize as a new DataVarConfig
            super().__init__(config, var_id, proj_dir)

        self.ts = None
        self.ts_type = None  # 'real' or 'surr'
        self.surr_num = None
        self.col_name = None
        self.time_var = None

        # self.pyleo_ts = None

    @property
    def ps(self):
        time_axis = np.abs(self.ts[self.time_var].values) # absolute time values for pyleo, imply direction via time_unit
        source_ps = pyleo.Series(time=time_axis, value=self.ts[self.col_name].values,
                                 time_unit=self.time_unit if self.time_unit is not None else 'yr BP', value_unit=self.unit, value_name=self.var,
                                 # label='wu_tsi')
                                 label=self.var_name)
        return source_ps


    def set_col_name(self):
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

        if ('time' not in df.columns) and (specified_time_var is not None):
            df = df.rename(columns={specified_time_var: 'time'})
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
        df['time'] = df['time'].astype('int')

        return df, 'time'


    def pull_ts(self, surr_num=None):
        if surr_num is None:
            self.get_real()
        else:
            self.surr_num = surr_num
            self.get_surr(surr_num=surr_num)


    def get_real(self):
        # get raw timeseries data from csv
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

