import yaml

from cedarkit.core.data_var import VarObject


def test_var_object_loads_project_level_variable_config_when_config_is_none(tmp_path):
    config_dir = tmp_path / "data_var_configs"
    config_dir.mkdir()
    (config_dir / "Example.yaml").write_text(
        yaml.safe_dump(
            {
                "Example": {
                    "var": "temperature",
                    "real_data_ts": {
                        "csv_stem": "example_real",
                        "var": "example_temperature",
                        "time": "time",
                    },
                    "surrogate_ts": {
                        "csv_stem": "example_surr",
                        "var": "temperature",
                        "time": "time",
                    },
                }
            }
        )
    )

    var_object = VarObject(None, "Example", tmp_path)

    assert var_object.real_ts_csv == "example_real"
    assert var_object.real_ts_var == "example_temperature"
    assert var_object.real_data_dir_path == tmp_path / "master_data"
    assert var_object.surr_data_dir_path == tmp_path / "master_surrogates"
    assert not var_object.real_data_dir_path.exists()
    assert not var_object.surr_data_dir_path.exists()


def test_var_object_derives_decadal_names_when_real_data_names_are_missing(tmp_path):
    config_dir = tmp_path / "data_var_configs"
    config_dir.mkdir()
    (config_dir / "Example.yaml").write_text(
        yaml.safe_dump(
            {
                "Example": {
                    "author": "Döring et al.",
                    "year": "2022",
                    "source": "GISP2",
                    "obs_type": "temp (d15N)",
                    "var": "temperature",
                    "surrogate_ts": {},
                }
            }
        )
    )

    var_object = VarObject(None, "Example", tmp_path, suffix_label="linear")
    var_object.ts_type = "real"
    var_object.set_col_name()

    assert var_object.real_ts_var == "Example_decavg_temperature"
    assert var_object.col_name == "Example_decavg_temperature__linear"
    assert var_object.real_csv_stem == "Doring_et_al_2022_GISP2_temp_d15N_decavg_temperature"
    assert var_object.real_ts_csv == "Doring_et_al_2022_GISP2_temp_d15N_decavg_temperature__linear"
