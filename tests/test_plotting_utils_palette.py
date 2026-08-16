import importlib.util
from pathlib import Path

import pyarrow as pa


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cedarkit"
    / "utils"
    / "plotting"
    / "plotting_utils.py"
)

SPEC = importlib.util.spec_from_file_location("plotting_utils_module", MODULE_PATH)
PLOTTING_UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOTTING_UTILS)

check_palette_syntax = PLOTTING_UTILS.check_palette_syntax


def test_arrow_relation_prefers_operation_aliases():
    table = pa.table({"relation": ["A -> B"]})
    palette = {
        "A reconstructs B": "green",
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A -> B"] == "green"


def test_arrow_relation_can_fall_back_to_reverse_causal_alias():
    table = pa.table({"relation": ["A -> B"]})
    palette = {
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A -> B"] == "blue"


def test_causal_relation_can_match_reverse_operation_alias():
    table = pa.table({"relation": ["A causes B"]})
    palette = {
        "B reconstructs A": "green",
        "A reconstructs B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A causes B"] == "green"


def test_reconstructs_relation_keeps_reverse_causal_alias():
    table = pa.table({"relation": ["A reconstructs B"]})
    palette = {
        "B causes A": "blue",
        "A causes B": "red",
    }

    resolved = check_palette_syntax(palette, table, default_color="gray")

    assert resolved["A reconstructs B"] == "blue"
