# cedarkit

`cedarkit` is a support package for CCM/EDM workflows. It provides the project-configuration loader, data-variable metadata objects, routing and path helpers, parquet packaging utilities, visualization helpers, and runner scripts used to move from project configs and parameter files to calculation outputs and aggregated artifacts.

This README is meant to help readers recognize the main files, directories, and output products in the current CSV/parquet-oriented workflow. SQLite-related paths and options are intentionally not documented here.

---

## Installation

The package metadata lives in `pyproject.toml` and currently targets Python 3.11+.

For development, install from the `cedar_util` directory with:

```bash
pip install -e .
```

There is also an optional Conda environment in `env.yml` for projects that want a fuller local environment around the package.

Core dependencies currently include:

- `pandas`
- `pyarrow`
- `matplotlib`
- `seaborn`
- `PyYAML`
- `cloudpickle`
- `joblib`
- `pyleoclim`
- `pyEDM`

For reproducible project environments, some workflows may pin `pyEDM` or related dependencies to specific commits outside the minimal package metadata.

---

## Repository Layout

```text
cedar_util/
├── cedarkit/
│   ├── core/          # ProjectConfig, data-variable objects, relationship/data objects
│   ├── utils/         # Routing, IO, workflow, CLI, tables, plotting helpers
│   └── viz/           # Grid/panel visualization helpers
├── runners/           # Script entrypoints for grouping, running, packaging, and grids
├── tests/             # Lightweight comparison / validation scripts
├── pyproject.toml     # Package metadata
├── requirements.txt   # Pinned environment-style dependency list
├── env.yml            # Optional Conda environment
└── README.md
```

---

## Workflow Overview

At a high level, the current documented workflow looks like this:

1. A project directory contains a `proj_config.yaml` plus variable YAMLs in `data_var_configs/`.
2. Parameter CSVs in a `parameters/` directory define the CCM runs to perform.
3. Runner scripts operate on those parameter or group files and write intermediate calculation outputs.
4. Packaging and aggregation steps turn those outputs into parquet tables and object-grid artifacts.

The important thing to recognize is the artifact flow:

```text
proj_config.yaml
data_var_configs/*.yaml
        ↓
parameters/*.csv
        ↓
calc_grps.csv / E_tau_grps.csv
        ↓
run-level output files
        ↓
parquet outputs and object-grid joblib files
```

---

## Configuration Model

The main config loader is `cedarkit.core.project_config.load_config`, which reads a top-level project YAML and can also load per-variable YAML files from `data_var_configs/`. The resulting `ProjectConfig` object exposes nested config sections as attributes.

Key config ideas reflected in the code:

- `data_vars`: names or IDs of variable definitions to load
- project-level metadata: labels, prefixes, run settings, file naming blocks
- `pal`: palette information, optionally supplemented by `data_var_configs/palette.yaml`
- location-specific settings: for example `local` and `hpc` blocks used by routing helpers
- variable metadata consumed by `DataVarConfig`, such as real/surrogate source fields and labels

A minimal fragment looks more like this:

```yaml
proj_name: ExampleProject
prefix: ExProj

data_vars:
  col: NGRIP1
  target: Wu18TSI

col:
  var_id: NGRIP1
  var: d18O

target:
  var_id: Wu18TSI
  var: TSI

csvs:
  calc_grps: calc_grps
  e_tau_grps: E_tau_grps
```

Typical companion files live under:

```text
proj_config.yaml
data_var_configs/
├── NGRIP1.yaml
├── Wu18TSI.yaml
└── palette.yaml
```

---

## Recognizing Common Input Files

Parameter and grouping files are plain CSVs. The exact columns vary by workflow, but these field names appear repeatedly in runners and output packaging:

```csv
col_var_id,target_var_id,E,tau,lag,knn,Tp,train_ind_i,id
NGRIP1,Wu18TSI,4,1,-5,4,0,0,101
```

Some files to look for:

- `parameters/*.csv`: run definitions consumed by calculation runners
- `calc_grps.csv`: grouped combinations used for packaging and aggregation
- `E_tau_grps.csv`: deduplicated `E`/`tau`-style grouping file written alongside calc groups

If you are scanning a project directory, the filenames are usually more informative than the command history.

---

## Runner Scripts

The scripts in `cedar_util/runners/` are the main operational entrypoints. They share a common CLI parser, so you will often see flags such as `--project`, `--config`, `--parameters`, `--inds`, `--proj_dir`, and `--test`, but the most useful thing here is knowing what each script reads and writes.

### `calc_groups.py`

Purpose: turn parameter CSVs into deduplicated group definitions.

Typical inputs:

- `proj_config.yaml`
- `parameters/*.csv`

Typical outputs:

- `calc_grps.csv`
- `E_tau_grps.csv`

Recognition snippet:

```text
parameters/params_lag_bycol_*.csv
calc_local_tmp/calc_grps.csv
calc_local_tmp/E_tau_grps.csv
```

### `run_ccm_experiments.py`

Purpose: execute CCM runs for rows in a parameter CSV using the loaded project config and write run-level output files.

Typical inputs:

- `proj_config.yaml`
- `parameters/<something>.csv`

Typical outputs:

- per-run files under the calculation directory chosen by config/routing

Recognition snippet:

```text
proj_config.yaml
parameters/params_*.csv
calc_local_tmp/
```

### `to_parquet.py`

Purpose: package grouped calculation outputs into parquet tables.

Typical inputs:

- `calc_grps.csv`
- `E_tau_grps.csv` or another grouping CSV
- intermediate run outputs from the calculation directory

Typical outputs:

- parquet outputs organized under the configured output directory structure

Recognition snippet:

```text
calc_grps.csv
E_tau_grps.csv
output/parquet/
```

### `object_grid.py`

Purpose: build grid-style aggregate objects and derived tables across `E`/`tau` combinations.

Typical inputs:

- grouped output artifacts
- project config metadata

Typical outputs:

- `delta_rho_stats` tables
- `delta_rho_full` tables
- `libsize_aggregated` tables
- object-grid joblib artifacts

Recognition snippet:

```text
E4_tau1__delta_rho_stats
E4_tau1__delta_rho_full
E4_tau1__libsize_aggregated
*.joblib
```

### `rewrite_object_grid_paths.py`

Purpose: rewrite embedded paths inside an existing object-grid joblib when moving the artifact to a new dyad/tmp home layout.

Typical inputs:

- an existing object-grid `.joblib`

Typical outputs:

- a rewritten `.joblib`, either in place or at a new path

Recognition snippet:

```text
--input-grid old_grid.joblib
--output-grid rewritten_grid.joblib
```

---

## Python Interfaces Worth Knowing

The README does not try to fully document the Python API, but these names are useful anchors when reading code:

- `cedarkit.core.project_config.load_config`
- `cedarkit.core.project_config.ProjectConfig`
- `cedarkit.core.data_var.DataVarConfig`

These interfaces are mostly useful for understanding how project YAML, variable YAML, and path-routing information are turned into runtime objects.

---

## Output Patterns

A few output patterns show up repeatedly across the package:

```text
proj_config.yaml
parameters/params_*.csv
calc_grps.csv
E_tau_grps.csv
E{E}_tau{tau}__delta_rho_stats
E{E}_tau{tau}__delta_rho_full
E{E}_tau{tau}__libsize_aggregated
*.parquet
*.joblib
```

If you are trying to identify the right file to inspect, these names are usually the quickest landmarks.

---

## Documentation Boundary

This README describes the current `cedarkit` CSV/parquet-oriented workflow and the artifacts most readers will encounter in this repository. Alternate or evolving paths in the codebase, including SQLite-related routing, are intentionally left out here.

---

## License

The current package metadata declares `GPL-3.0`.
