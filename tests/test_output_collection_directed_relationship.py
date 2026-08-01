from types import SimpleNamespace

import pandas as pd
import pytest

from cedarkit.core.data_objects import Output, OutputCollection
from cedarkit.viz.panels import BasePlot, LagPlot


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
