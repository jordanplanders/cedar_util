
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Mapping, Any
from scipy.interpolate import interp1d

# -----------------------------
# Data Variable Object (minimal)
# -----------------------------
@dataclass(frozen=True)
class DataVariable:
    """
    Minimal descriptor for a dataset variable.
    """
    var: str                  # human-facing variable key, e.g., "TSI" or "temp"
    var_id: str               # unique id string you use elsewhere
    data_var_name: str        # column name in source data
    surr_csv: Optional[str]   # path/uri to surrogate csv (if any)
    unit: str                 # unit string, e.g., "W/m^2"
    label: str                # short label for plotting/tables
    age_model_csv: Optional[str]  # path/uri to age-model info (if any)
    source: str               # data source name or citation key


# Time Series Configuration Object
# attributes: name, name_surr, name_age_model, name_surr_age_model, surr_flag, surr_num, age_model_id, palette
# methods: apply_age_model(), resample()

# ---------------------------------
# Time Series Configuration (minimal)
# ---------------------------------
@dataclass(frozen=True)
class TimeSeriesConfig:
    """
    Minimal configuration describing names/flags and rendering choices.
    """
    # attributes
    name: str
    name_surr: str
    name_age_model: str
    name_surr_age_model: str
    surr_flag: bool
    surr_num: Optional[int]
    age_model_id: Optional[str]
    palette: Any  # keep open: could be list[str] or dict[str, str]
    df: Optional[pd.DataFrame] = None  # placeholder for actual data


    # methods
    def _get_age_model(self):
        pass

    def apply_age_model(self, sampling_times, target_time_series):
        alt_age_axis = self._get_age_axis()


        f = interp1d(
            x=alt_age_axis,
            y=target_time_series, fill_value='extrapolate')

        return pd.DataFrame({time_var: sampling_times, 'value':f(sampling_times)})

