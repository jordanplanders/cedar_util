from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _import_relationship_module():
    module_path = Path(__file__).resolve().parents[1] / "cedarkit" / "core" / "relationship.py"
    spec = spec_from_file_location("relationship_under_test", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_relationship_module = _import_relationship_module()
Relationship = _relationship_module.Relationship
RelationshipSide = _relationship_module.RelationshipSide


def test_relationship_pres_channel_preserves_r1_r2_direction():
    relationship = Relationship(var_x="temp", var_y="TSI")

    assert relationship.r1 == "TSI causes temp"
    assert relationship.r2 == "temp causes TSI"


def test_relationship_calc_channel_uses_operation_direction():
    relationship = Relationship(var_x="temp", var_y="TSI", output_convention="operation")

    assert relationship.r1_calc == "temp reconstructs TSI"
    assert relationship.r2_calc == "TSI reconstructs temp"


def test_relationship_mappings_round_trip_between_calc_and_pres():
    relationship = Relationship(var_x="temp", var_y="TSI", output_convention="operation")

    assert relationship.to_calc_mapping["TSI causes temp"] == "temp reconstructs TSI"
    assert relationship.to_pres_mapping["temp reconstructs TSI"] == "TSI causes temp"
    assert relationship.to_pres_mapping["temp (surr) -> TSI"] == "TSI causes temp (surr)"


def test_relationship_side_supports_written_calc_mapping():
    side = RelationshipSide(
        "r1",
        var_x="temp",
        var_y="TSI",
        output_convention="operation",
        convention_mapping={"reconstructs": "->"},
    )

    assert side.r_calc == "temp -> TSI"
    assert side.to_pres_mapping["temp -> TSI"] == "TSI causes temp"


def test_surrogate_calc_variant_tracks_operation_channel():
    relationship = Relationship(var_x="temp", var_y="TSI", output_convention="operation")

    assert relationship.surr_r1x_calc == "temp (surr) reconstructs TSI"


def test_relation_aliases_identify_generic_r1_category_only():
    relationship = Relationship(var_x="E", var_y="F", output_convention="operation")

    aliases = relationship.relation_aliases("r1")

    assert {
        "E reconstructs F",
        "E -> F",
        "F causes E",
        "F influences E",
    }.issubset(aliases)
    assert all("(surr)" not in alias for alias in aliases)


def test_relation_aliases_identify_generic_r2_category_only():
    relationship = Relationship(var_x="E", var_y="F", output_convention="operation")

    aliases = relationship.relation_aliases("r2")

    assert {
        "F reconstructs E",
        "F -> E",
        "E causes F",
        "E influences F",
    }.issubset(aliases)
