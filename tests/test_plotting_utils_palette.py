import pandas as pd

from cedarkit.utils.plotting.plotting_utils import check_palette_syntax


def test_relation_prefers_operation_aliases():
    table = pd.DataFrame({"relation": ["A -> B"]})
    palette = {
        "A reconstructs B": "green",
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A -> B"] == "green"


def test_relation_can_fall_back_to_reverse_causal_alias():
    table = pd.DataFrame({"relation": ["A -> B"]})
    palette = {
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A -> B"] == "blue"


def test_causal_relation_can_match_reverse_operation_alias():
    table = pd.DataFrame({"relation": ["A causes B"]})
    palette = {
        "B reconstructs A": "green",
        "A reconstructs B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A causes B"] == "green"


def test_reconstructs_relation_keeps_reverse_causal_alias():
    table = pd.DataFrame({"relation": ["A reconstructs B"]})
    palette = {
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A reconstructs B"] == "blue"
