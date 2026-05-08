import logging

logger = logging.getLogger(__name__)
try:
    from cedarkit.utils.cli import log_line
except ImportError:
    def log_line(*args, **kwargs):
        return None


class RelationshipSide:
    """Render one directional relationship side in calc and presentation conventions.

    Directionality is fixed by `r` and must not be altered by convention changes:

    - `r1`: calc is `x reconstructs y`, pres is `y causes x`
    - `r2`: calc is `y reconstructs x`, pres is `x causes y`
    """

    def __init__(
        self,
        r,
        relationship=None,
        var_x="temp",
        var_y="TSI",
        influence_word="causes",
        operation_word="reconstructs",
        output_convention="influence",
        pres_convention="influence",
        convention_mapping=None,
        output_string=None,
    ):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.var_x = var_x if relationship is None else relationship.var_x
        self.var_y = var_y if relationship is None else relationship.var_y
        self.influence_word = influence_word
        self.operation_word = operation_word
        self.output_convention = output_convention
        self.pres_convention = pres_convention
        self.convention_mapping = dict(convention_mapping or {})
        self.output_string = output_string

        self.surr_rx_count = None
        self.surr_rx_count_outperforming = None
        self.surr_ry_count = None
        self.surr_ry_count_outperforming = None
        self.delta_rho = None
        self.maxlibsize_rho = None
        self.lag = None
        self.surr_rx_outperforming_frac = None
        self.surr_ry_outperforming_frac = None
        self.peak_start = None
        self.peak_end = None

        # Direction evidence: r1 calc must stay x reconstructs y; pres must stay y causes x.
        if r == "r1":
            self.pattern_calc = "x reconstructs y" if self.output_convention == "operation" else "y causes x"
            self.pattern_pres = "x reconstructs y" if self.pres_convention == "operation" else "y causes x"
        # Direction evidence: r2 calc must stay y reconstructs x; pres must stay x causes y.
        elif r == "r2":
            self.pattern_calc = "y reconstructs x" if self.output_convention == "operation" else "x causes y"
            self.pattern_pres = "y reconstructs x" if self.pres_convention == "operation" else "x causes y"
        else:
            raise ValueError(f"Unknown relationship side {r!r}")

        self.r_id = r

    def _render(self, pattern, x_value, y_value, *, use_mapping=True):
        text = pattern.replace("x", x_value).replace("y", y_value)
        influence_text = self.influence_word
        operation_text = self.operation_word
        if use_mapping:
            influence_text = self.convention_mapping.get(influence_text, influence_text)
            operation_text = self.convention_mapping.get(operation_text, operation_text)
        return text.replace("causes", influence_text).replace("reconstructs", operation_text)

    def _mapping_variants(self, pattern, x_value, y_value):
        variants = {
            self._render(pattern, x_value, y_value, use_mapping=False),
            self._render(pattern, x_value, y_value, use_mapping=True),
        }
        return {v for v in variants if v}

    @property
    def surr_rx(self):
        return self._render(self.pattern_pres, f"{self.var_x} (surr)", self.var_y)

    @property
    def surr_rx_calc(self):
        return self._render(self.pattern_calc, f"{self.var_x} (surr)", self.var_y)

    @property
    def surr_ry(self):
        return self._render(self.pattern_pres, self.var_x, f"{self.var_y} (surr)")

    @property
    def surr_ry_calc(self):
        return self._render(self.pattern_calc, self.var_x, f"{self.var_y} (surr)")

    @property
    def r(self):
        return self._render(self.pattern_pres, self.var_x, self.var_y)

    @property
    def r_calc(self):
        return self._render(self.pattern_calc, self.var_x, self.var_y)

    @property
    def to_calc_mapping(self):
        target_main = self.r_calc
        target_surr_rx = self.surr_rx_calc
        target_surr_ry = self.surr_ry_calc
        mapping = {}
        for relation in self._mapping_variants(self.pattern_calc, self.var_x, self.var_y) | self._mapping_variants(
            self.pattern_pres, self.var_x, self.var_y
        ):
            mapping[relation] = target_main
        for relation in self._mapping_variants(self.pattern_calc, f"{self.var_x} (surr)", self.var_y) | self._mapping_variants(
            self.pattern_pres, f"{self.var_x} (surr)", self.var_y
        ):
            mapping[relation] = target_surr_rx
        for relation in (
            self._mapping_variants(self.pattern_calc, self.var_x, f"{self.var_y} (surr)")
            | self._mapping_variants(self.pattern_pres, self.var_x, f"{self.var_y} (surr)")
        ):
            mapping[relation] = target_surr_ry
        return mapping

    @property
    def to_pres_mapping(self):
        target_main = self.r
        target_surr_rx = self.surr_rx
        target_surr_ry = self.surr_ry
        mapping = {}
        for relation in self._mapping_variants(self.pattern_calc, self.var_x, self.var_y) | self._mapping_variants(
            self.pattern_pres, self.var_x, self.var_y
        ):
            mapping[relation] = target_main
        for relation in self._mapping_variants(self.pattern_calc, f"{self.var_x} (surr)", self.var_y) | self._mapping_variants(
            self.pattern_pres, f"{self.var_x} (surr)", self.var_y
        ):
            mapping[relation] = target_surr_rx
        for relation in (
            self._mapping_variants(self.pattern_calc, self.var_x, f"{self.var_y} (surr)")
            | self._mapping_variants(self.pattern_pres, self.var_x, f"{self.var_y} (surr)")
        ):
            mapping[relation] = target_surr_ry
        return mapping


class Relationship:
    """Relationship family with calc-facing and presentation-facing render channels."""

    def __init__(
        self,
        var_x="temp",
        var_y="TSI",
        surr_flag="neither",
        influence_word="causes",
        operation_word="reconstructs",
        output_convention="influence",
        pres_convention="influence",
        convention_mapping=None,
    ):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.influence_word = influence_word
        self.var_x = var_x
        self.var_y = var_y
        self.surr_flag = surr_flag
        self.operation_word = operation_word
        self.output_convention = output_convention
        self.pres_convention = pres_convention
        self.convention_mapping = dict(convention_mapping or {})

        self._r1_side = RelationshipSide(
            "r1",
            relationship=self,
            influence_word=self.influence_word,
            operation_word=self.operation_word,
            output_convention=self.output_convention,
            pres_convention=self.pres_convention,
            convention_mapping=self.convention_mapping,
        )
        self._r2_side = RelationshipSide(
            "r2",
            relationship=self,
            influence_word=self.influence_word,
            operation_word=self.operation_word,
            output_convention=self.output_convention,
            pres_convention=self.pres_convention,
            convention_mapping=self.convention_mapping,
        )

        self.pattern_calc = {"r1": self._r1_side.pattern_calc, "r2": self._r2_side.pattern_calc}
        self.pattern_pres = {"r1": self._r1_side.pattern_pres, "r2": self._r2_side.pattern_pres}

    def set_influence_verb(self, verb):
        self.influence_word = verb
        self._r1_side.influence_word = verb
        self._r2_side.influence_word = verb

    def set_operation_verb(self, verb):
        self.operation_word = verb
        self._r1_side.operation_word = verb
        self._r2_side.operation_word = verb

    def set_active_r1(self):
        if self.surr_flag in ("x", self.var_x):
            return self.surr_r1x
        elif self.surr_flag in ("neither",):
            return self.r1
        elif self.surr_flag in ("y", self.var_y):
            return self.surr_r1y
        elif self.surr_flag in ("both",):
            return self.surr_r1yx

    def set_active_r2(self):
        if self.surr_flag in ("x", self.var_x):
            return self.surr_r2x
        elif self.surr_flag in ("neither",):
            return self.r2
        elif self.surr_flag in ("y", self.var_y):
            return self.surr_r2y
        elif self.surr_flag in ("both",):
            return self.surr_r2yx

    @property
    def r1(self):
        return self._r1_side.r

    @property
    def r1_calc(self):
        return self._r1_side.r_calc

    @property
    def r2(self):
        return self._r2_side.r

    @property
    def r2_calc(self):
        return self._r2_side.r_calc

    @property
    def surr_r1x(self):
        return self._r1_side.surr_rx

    @property
    def surr_r1x_calc(self):
        return self._r1_side.surr_rx_calc

    @property
    def surr_r1y(self):
        return self._r1_side.surr_ry

    @property
    def surr_r1y_calc(self):
        return self._r1_side.surr_ry_calc

    @property
    def surr_r2x(self):
        return self._r2_side.surr_rx

    @property
    def surr_r2x_calc(self):
        return self._r2_side.surr_rx_calc

    @property
    def surr_r2y(self):
        return self._r2_side.surr_ry

    @property
    def surr_r2y_calc(self):
        return self._r2_side.surr_ry_calc

    @property
    def surr_r2xy(self):
        # TODO: review for removal post-Part1A; currently matches surr_r2yx exactly.
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r2yx(self):
        # TODO: review for removal post-Part1A; currently matches surr_r2xy exactly.
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r2both(self):
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r1xy(self):
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def surr_r1yx(self):
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def surr_r1both(self):
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def to_calc_mapping(self):
        mapping = {}
        mapping.update(self._r1_side.to_calc_mapping)
        mapping.update(self._r2_side.to_calc_mapping)
        return mapping

    @property
    def to_pres_mapping(self):
        mapping = {}
        mapping.update(self._r1_side.to_pres_mapping)
        mapping.update(self._r2_side.to_pres_mapping)
        return mapping
