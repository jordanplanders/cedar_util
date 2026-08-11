from pathlib import Path

import pytest
import yaml

from cedarkit.core.project_config import ProjectConfig, load_config


def test_project_dir_is_resolved_from_source_config_path(tmp_path):
    config_path = tmp_path / "project" / "proj_config.yaml"
    config = ProjectConfig({}, file_path=config_path)

    assert config.project_dir == config_path.resolve().parent


def test_project_dir_requires_source_config_path():
    config = ProjectConfig({})

    with pytest.raises(ValueError, match="file_path"):
        _ = config.project_dir


def test_patch_source_preserves_external_variable_structure(tmp_path):
    config_path = tmp_path / "proj_config.yaml"
    variable_dir = tmp_path / "data_var_configs"
    variable_dir.mkdir()
    config_path.write_text(yaml.safe_dump({"data_vars": {"A": "A"}, "existing": 1}))
    (variable_dir / "A.yaml").write_text(
        yaml.safe_dump({"A": {"real_data_ts": {"csv_stem": "a"}}})
    )
    config = load_config(config_path)

    config.patch_source({"time_axis.csv": "axis", "time_axis.t0": 0.0})

    source = yaml.safe_load(config_path.read_text())
    assert source["data_vars"] == {"A": "A"}
    assert "A" not in source
    assert source["existing"] == 1
    assert source["time_axis"] == {"csv": "axis", "t0": 0.0}
