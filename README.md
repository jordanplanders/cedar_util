# cedar

Tools for managing configuration, parameter generation, data-object handling, and HPC calculations for CCM/EDM workflows. This codebase supports the full parameter-sweep pipeline used in Jordan Landers’ CCM analyses (E–τ grids, lags, surrogates, parquet conversion, config parsing, and data‑var metadata).

This repository provides:
- a small library of core utilities (`utils/`) for coordination tasks, file handling, configuration logic, and output preparation  
- data‑structure and metadata classes (`data_obj/`)  
- HPC scripts for packaging and recovering calculations (`hpc/`)  
- local workflow tools for generating parameters and grouping runs (`local2/`)  

Analysis environments and CCM computations live in separate projects; `cedar` supplies the supporting infrastructure.

---

## Installation

`cedar` uses a `pyproject.toml` with a minimal dependency set.  
One dependency—`pyEDM`—is installed from a project-specific fork and pinned to a
specific commit for reproducibility.

Install with:

```bash
pip install .
```

or for editable development:

```bash
pip install -e .
```

### Dependencies (automatically installed)

- pandas  
- pyarrow  
- pyleoclim  
- matplotlib  
- seaborn  
- PyYAML  
- cloudpickle  
- joblib  
- pyEDM (from a project-specific fork, pinned to commit `5defad56b...`)  

---

## Repository Structure

```
cedar/
│
├── data_obj/           # DataVar, Relationship, and plotting-related classes
│
├── utils/              # Helpers for coordination, file handling, configs,
│                       # and post-processing
│
├── local2/             # Local workflow tools (make params, group calculations)
│
├── hpc/                # CARC / HPC pipeline scripts and recovery tools
│
├── pyproject.toml      # Package metadata and dependencies
│
└── README.md           # (this file)
```

---

## Typical Usage

`cedar` is designed to be imported by external scripts,
notebooks, and HPC pipelines. Examples:

```python
from cedar.utils.config_parser import ProjectConfig
from cedar.data_obj.data_var import DataVar
from cedar.utils.process_output import OutputProcessor
```

### Local workflow utilities

Group parameter files into calculation bundles:

```bash
python -m cedar.local2.calc_grps --project myproj
```

### HPC usage

CARC-run scripts are located in `cedar/hpc/`:

- `to_parquet.py`  
- `run_edm_carc_pool2_config_toObj.py`  

These are designed to be called from SLURM job scripts with
project-specific configs and parameter-group files.

---

## Configuration Files

Projects define variables and metadata using YAML configurations parsed by
`ProjectConfig`. Example:

```yaml
proj_name: NGRIP1Seierstad14d18OLinear_Wu18TSILinear 
prefix: SeNGLWuL
doi: 10.6084/m9.figshare.30632639

data_vars:
  NGRIP1Seierstad14d18OLinear: NGRIP1Seierstad14d18OLinear
  Wu18TSILinear : Wu18TSILinear

col:
  var_id: NGRIP1Seierstad14d18OLinear
  var: d18O
  long_label: d18O
target:
  var_id: Wu18TSILinear
  var: TSI
  long_label: Total Solar Irradiance

...
```

The parser turns nested dictionaries into attribute-ready objects, keeping
project definitions clean and reproducible.

---

## Development Notes

- The pinned pyEDM fork contains custom CCM/EDM modifications used by the
  calculation pipeline.
- Module structure aims to remain stable and easy to extend. Future additions
  (e.g., refinements to data objects or output schema) should maintain
  backward compatibility with `ProjectConfig` and output-handling tools.

---

## License

MIT (or choose your preferred license).
