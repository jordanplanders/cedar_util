from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from cedarkit.core.data_objects import Output, OutputCollection, RunConfig
from cedarkit.viz.panels import BasePlot, LagPlot


def test_output_rebases_a_copied_hpc_path_from_its_local_dyad(tmp_path):
    dyad_dir = tmp_path / "my_project" / "dyads" / "A_B"
    local_file = dyad_dir / "tmp" / "result.parquet"
    local_file.parent.mkdir(parents=True)
    pl.DataFrame({"rho": [0.25]}).write_parquet(local_file)

    old_hpc_path = Path("/hpc/archive/my_project/dyads/A_B/tmp/result.parquet")
    output = Output(None, path=old_hpc_path, tmp_dir=dyad_dir / "tmp")

    assert output.table.collect()["rho"].to_list() == [0.25]
    assert output.path == local_file


def test_collection_rebases_all_paths_using_the_explicit_dyad(tmp_path):
    dyad_dir = tmp_path / "my_project" / "multi_player" / "A_B"
    local_file = dyad_dir / "nested" / "result.parquet"
    local_file.parent.mkdir(parents=True)
    local_file.touch()

    old_hpc_path = Path("/hpc/archive/my_project/multi_player/A_B/nested/result.parquet")
    collection = OutputCollection(tmp_dir=dyad_dir / "tmp")
    collection.delta_rho_stats = Output(None, path=old_hpc_path)
    collection.grp_config = RunConfig({}, tmp_dir=dyad_dir / "tmp")
    collection.grp_config.output_path = [old_hpc_path]

    assert collection.resolve_paths(dyad_dir) == 2
    assert collection.delta_rho_stats.path == local_file
    assert collection.grp_config.output_path == [local_file]


def test_collection_prefers_local_files_over_a_reachable_saved_path(tmp_path):
    dyad_dir = tmp_path / "my_project" / "dyads" / "A_B"
    local_file = dyad_dir / "tmp" / "result.parquet"
    local_file.parent.mkdir(parents=True)
    local_file.touch()
    mounted_hpc_file = tmp_path / "mounted_hpc" / "A_B" / "tmp" / "result.parquet"
    mounted_hpc_file.parent.mkdir(parents=True)
    mounted_hpc_file.touch()

    collection = OutputCollection(tmp_dir=dyad_dir / "tmp")
    collection.delta_rho_stats = Output(None, path=mounted_hpc_file)

    assert collection.resolve_paths(dyad_dir) == 1
    assert collection.delta_rho_stats.path == local_file


def test_run_config_pull_output_filters_parquet_with_polars(tmp_path):
    output_file = tmp_path / "output.parquet"
    pl.DataFrame(
        {
            "E": [3, 4],
            "tau": [1, 1],
            "rho": [0.1, 0.2],
        }
    ).write_parquet(output_file)
    run_config = RunConfig({"E": 4, "tau": 1})
    run_config.output_path = [output_file]

    result = run_config.pull_output(to_table=True)

    assert isinstance(result, pl.DataFrame)
    assert result.to_dicts() == [{"E": 4, "tau": 1, "rho": 0.2}]


def test_supplied_directed_relationship_is_the_only_active_side(tmp_path):
    relationship = SimpleNamespace(
        r="B influences A",
        r_calc="B reconstructs A",
        participant_variables=("B", "A"),
    )
    collection = OutputCollection(tmp_dir=tmp_path)

    collection.set_relationships(relationship=relationship)

    assert collection.relationships is relationship
    assert collection.r1 is relationship
    assert collection.get_relationship("r1") is relationship
    assert collection.r2 is None
    with pytest.raises(ValueError, match="no r2"):
        collection.get_relationship("r2")


def test_default_metric_calculation_only_attempts_r1_for_directed_relationship(tmp_path):
    relationship = SimpleNamespace(
        r="B influences A",
        r_calc="B reconstructs A",
        participant_variables=("B", "A"),
    )
    collection = OutputCollection(tmp_dir=tmp_path)
    collection.set_relationships(relationship=relationship)
    calls = []
    collection.delta_rho_stats = SimpleNamespace(get_table=lambda: None, clear_table=lambda: None)
    collection._calc_metrics = lambda relationship_id, **_: calls.append(relationship_id)

    collection.calc_metrics()

    assert calls == ["r1"]


def test_pull_df_concatenates_directed_collections_with_comparison_labels(tmp_path):
    def collection(value):
        output = Output(
            pd.DataFrame(
                {
                    "lag": [0],
                    "rho": [value],
                    "relation": ["B reconstructs A"],
                    "relation_0": ["B reconstructs A"],
                    "surr_var": ["neither"],
                    "surr_num": [0],
                }
            ),
            outtype="delta_rho_stats",
            tmp_dir=tmp_path,
        )
        result = OutputCollection(tmp_dir=tmp_path)
        result.delta_rho_stats = output
        result.relationships = SimpleNamespace(
            to_pres_mapping={"B reconstructs A": "A influences B"},
            to_calc_mapping={"B reconstructs A": "B reconstructs A"},
        )
        return result

    plot = BasePlot({"palette": {}})
    result = plot.pull_df(
        [collection(0.2), collection(0.4)],
        "delta_rho_stats",
        "full",
        columns=["lag", "rho", "relation", "comparison_label"],
        comparison_labels=["exp", "equal"],
    )

    assert result["rho"].tolist() == [0.2, 0.4]
    assert result["relation"].tolist() == ["A influences B", "A influences B"]
    assert result["comparison_label"].tolist() == ["exp", "equal"]


def test_lag_plot_accepts_a_collection_list(tmp_path):
    def collection(value):
        result = OutputCollection(tmp_dir=tmp_path)
        result.delta_rho_stats = Output(
            pd.DataFrame(
                {
                    "lag": [0],
                    "delta_rho": [value],
                    "relation": ["B reconstructs A"],
                    "relation_0": ["B reconstructs A"],
                    "surr_var": ["neither"],
                    "surr_num": [0],
                }
            ),
            outtype="delta_rho_stats",
            tmp_dir=tmp_path,
        )
        result.relationships = SimpleNamespace(
            to_pres_mapping={"B reconstructs A": "A influences B"},
            to_calc_mapping={"B reconstructs A": "B reconstructs A"},
        )
        return result

    plot = LagPlot(palette={})
    axis = plot.make_classic_lag_plot(
        [collection(0.2), collection(0.4)],
        stats_only=True,
        scatter=False,
        comparison_labels=["exp", "equal"],
        hue="comparison_label",
    )

    assert axis is plot.ax
    assert len(axis.lines) > 0
