from types import SimpleNamespace

import pandas as pd
import polars as pl

from cedarkit.utils.io.parquet import package_calc_grp_results_to_parquet


def _config():
    return SimpleNamespace(
        col=SimpleNamespace(var="temp", var_id="temp_id"),
        target=SimpleNamespace(var="TSI", var_id="tsi_id"),
        output=SimpleNamespace(
            parquet=SimpleNamespace(file_format="E{E}_tau{tau}_lag{lag}")
        ),
    )


def _write_csv(path, *, relation, status=None):
    frame = pd.DataFrame({"rho": [0.2], "LibSize": [20], "relation": [relation]})
    if status is not None:
        frame["status"] = status
    frame.to_csv(path, index=False)


def test_package_results_uses_polars_for_write_skip_and_schema_evolution(tmp_path):
    source = tmp_path / "source"
    lag_dir = source / "lag0"
    lag_dir.mkdir(parents=True)
    destination = tmp_path / "parquet"
    parts = {"E": 4, "tau": 1, "Tp": 1, "knn": 20}

    first = lag_dir / "1_E4_tau1_lag_0__neither0.csv"
    _write_csv(first, relation="TSI causes temp")

    write_paths, existing = package_calc_grp_results_to_parquet(
        source, destination, parts, "temp", "TSI", _config()
    )
    assert existing == []
    assert len(write_paths) == 1

    output_path = write_paths[0]
    initial = pl.read_parquet(output_path)
    assert initial.height == 1
    assert initial["surr_var"].to_list() == ["neither"]

    write_paths, existing = package_calc_grp_results_to_parquet(
        source, destination, parts, "temp", "TSI", _config()
    )
    assert output_path in existing
    assert write_paths == []
    assert pl.read_parquet(output_path).height == 1

    second = lag_dir / "2_E4_tau1_lag_0__temp1.csv"
    _write_csv(second, relation="TSI causes temp", status="complete")
    package_calc_grp_results_to_parquet(source, destination, parts, "temp", "TSI", _config())

    combined = pl.read_parquet(output_path)
    assert combined.height == 2
    assert set(combined["surr_var"].to_list()) == {"neither", "temp"}
    assert "status" in combined.columns
    assert combined.filter(pl.col("surr_var") == "neither")["status"].null_count() == 1
