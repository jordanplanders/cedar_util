import pandas as pd

from cedarkit.core.data_objects import Output


def test_output_adds_surrogate_specific_relation_spec():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["E reconstructs F"] * 4,
                "surr_var": ["neither", "E", "F", "both"],
                "x_var": ["E"] * 4,
                "y_var": ["F"] * 4,
            }
        )
    )

    result = output.full.collect()

    assert result["relation"].to_list() == ["E reconstructs F"] * 4
    assert result["relation_spec"].to_list() == [
        "E reconstructs F",
        "E (surr) reconstructs F",
        "E reconstructs F (surr)",
        "E (surr) reconstructs F (surr)",
    ]


def test_output_derives_relation_spec_without_x_or_y_columns():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["E causes F"] * 4,
                "surr_var": ["neither", "E", "F", "both"],
            }
        )
    )

    result = output.full.collect()

    assert result["relation"].to_list() == ["E causes F"] * 4
    assert result["relation_spec"].to_list() == [
        "E causes F",
        "E (surr) causes F",
        "E causes F (surr)",
        "E (surr) causes F (surr)",
    ]


def test_output_uses_variable_columns_for_an_unparsed_relation_spelling():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["E drives F"],
                "surr_var": ["E"],
                "x_var": ["E"],
                "y_var": ["F"],
            }
        )
    )

    assert output.full.collect()["relation_spec"].to_list() == ["E (surr) drives F"]


def test_output_marks_the_named_surrogate_in_a_reversed_causal_relation():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["TSI causes temp", "TSI causes temp"],
                "surr_var": ["temp", "TSI"],
                "x_var": ["temp", "temp"],
                "y_var": ["TSI", "TSI"],
            }
        )
    )

    assert output.full.collect()["relation_spec"].to_list() == [
        "TSI causes temp (surr)",
        "TSI (surr) causes temp",
    ]


def test_output_replaces_a_stale_relation_spec_from_the_source_columns():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["E -> F"],
                "relation_spec": ["stale"],
                "surr_var": ["E"],
            }
        )
    )

    assert output.full.collect()["relation_spec"].to_list() == ["E (surr) -> F"]


def test_output_preserves_legacy_suffixed_relation_as_spec():
    output = Output(
        pd.DataFrame(
            {
                "relation": ["E (surr) reconstructs F"],
                "surr_var": ["E"],
                "x_var": ["E"],
                "y_var": ["F"],
            }
        )
    )

    result = output.full.collect()

    assert result["relation"].to_list() == ["E (surr) reconstructs F"]
    assert result["relation_spec"].to_list() == ["E (surr) reconstructs F"]
