import logging

logger = logging.getLogger(__name__)
from cedarkit.utils.cli import log_line


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
        """Build one side (``r1`` or ``r2``) of a directional relationship.

        Parameters
        ----------
        r : {'r1', 'r2'}
            Which side this is. Fixes the calc/pres sentence direction per
            the class docstring's invariant — ``r1`` calc is always
            "x reconstructs y" / pres is always "y causes x", and vice versa
            for ``r2``. Any other value raises ``ValueError``.
        relationship : Relationship, optional
            If given, ``var_x``/``var_y`` are taken from it instead of the
            ``var_x``/``var_y`` arguments below.
        var_x : str, optional
            Name of the "x" variable. Ignored if ``relationship`` is given.
            Default is ``"temp"``.
        var_y : str, optional
            Name of the "y" variable. Ignored if ``relationship`` is given.
            Default is ``"TSI"``.
        influence_word : str, optional
            Verb substituted for "causes" in pres-convention sentences.
            Default is ``"causes"``.
        operation_word : str, optional
            Verb substituted for "reconstructs" in calc-convention sentences.
            Default is ``"reconstructs"``.
        output_convention : {'operation', 'influence'}, optional
            Which sentence form (`pattern_calc`) describes this side's
            primary calc output. Default is ``"influence"``.
        pres_convention : {'operation', 'influence'}, optional
            Which sentence form (`pattern_pres`) describes this side's
            presentation-facing output. Default is ``"influence"``.
        convention_mapping : dict, optional
            Optional remapping applied to ``influence_word``/``operation_word``
            when rendering (see :meth:`_render`). Default is ``{}``.
        output_string : str, optional
            Currently stored as-is but not otherwise used by this class.

        Raises
        ------
        ValueError
            If ``r`` is not ``'r1'`` or ``'r2'``.
        """
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
        # Substitutes x_value/y_value into pattern, then substitutes
        # influence_word/operation_word for the literal "causes"/"reconstructs"
        # (optionally passed through convention_mapping first). Returns the rendered string.
        text = pattern.replace("x", x_value).replace("y", y_value)
        influence_text = self.influence_word
        operation_text = self.operation_word
        if use_mapping:
            influence_text = self.convention_mapping.get(influence_text, influence_text)
            operation_text = self.convention_mapping.get(operation_text, operation_text)
        return text.replace("causes", influence_text).replace("reconstructs", operation_text)

    def _mapping_variants(self, pattern, x_value, y_value):
        # Returns the set of distinct non-empty strings produced by rendering
        # pattern with and without convention_mapping applied (used to build
        # the to_calc_mapping/to_pres_mapping lookup tables below).
        variants = {
            self._render(pattern, x_value, y_value, use_mapping=False),
            self._render(pattern, x_value, y_value, use_mapping=True),
        }
        return {v for v in variants if v}

    def _relation_variants(self, x_value, y_value):
        """Return written and symbolic spellings for one directional variant."""
        variants = self._mapping_variants(self.pattern_calc, x_value, y_value) | self._mapping_variants(
            self.pattern_pres, x_value, y_value
        )
        if self.r_id == "r1":
            reconstructed, reconstructor = x_value, y_value
        else:
            reconstructed, reconstructor = y_value, x_value
        variants.update(
            {
                f"{reconstructed} -> {reconstructor}",
                f"{reconstructed} => {reconstructor}",
                f"{reconstructed} → {reconstructor}",
            }
        )
        return variants

    def relation_aliases(self):
        """Return unsuffixed spellings that identify this directional category.

        These aliases are for matching the generic ``relation`` column.  They
        intentionally exclude surrogate-specific forms; those are represented
        separately by ``surr_var`` and the derived ``relation_spec`` column.
        """
        if self.r_id == "r1":
            reconstructed, reconstructor = self.var_x, self.var_y
        else:
            reconstructed, reconstructor = self.var_y, self.var_x

        operation_words = {
            "reconstructs",
            self.operation_word,
            self.convention_mapping.get(self.operation_word, self.operation_word),
        }
        influence_words = {
            "causes",
            "influences",
            self.influence_word,
            self.convention_mapping.get(self.influence_word, self.influence_word),
        }
        aliases = {
            self.r_calc,
            self.r,
            f"{reconstructed} -> {reconstructor}",
            f"{reconstructed} => {reconstructor}",
            f"{reconstructed} → {reconstructor}",
        }
        aliases.update(
            f"{reconstructed} {word} {reconstructor}"
            for word in operation_words
            if word
        )
        aliases.update(
            f"{reconstructor} {word} {reconstructed}"
            for word in influence_words
            if word
        )
        return aliases

    @property
    def r(self):
        """This side's presentation-facing sentence, e.g. ``'y causes x'``.

        All ``r*``/``surr_*`` properties on this class are renderings of
        either ``self.pattern_pres`` (presentation convention) or
        ``self.pattern_calc`` (calc convention) via :meth:`_render`, with
        ``var_x``/``var_y`` swapped for a ``"(surr)"``-suffixed variant on
        the surrogate-flavored properties. See the class docstring for the
        fixed ``r1``/``r2`` directionality this always respects.

        Returns
        -------
        str
        """
        return self._render(self.pattern_pres, self.var_x, self.var_y)

    @property
    def r_calc(self):
        # Calc-convention counterpart of self.r (renders self.pattern_calc instead).
        return self._render(self.pattern_calc, self.var_x, self.var_y)

    @property
    def surr_rx(self):
        # Presentation-convention sentence with var_x replaced by "var_x (surr)".
        return self._render(self.pattern_pres, f"{self.var_x} (surr)", self.var_y)

    @property
    def surr_rx_calc(self):
        # Calc-convention counterpart of surr_rx.
        return self._render(self.pattern_calc, f"{self.var_x} (surr)", self.var_y)

    @property
    def surr_ry(self):
        # Presentation-convention sentence with var_y replaced by "var_y (surr)".
        return self._render(self.pattern_pres, self.var_x, f"{self.var_y} (surr)")

    @property
    def surr_ry_calc(self):
        # Calc-convention counterpart of surr_ry.
        return self._render(self.pattern_calc, self.var_x, f"{self.var_y} (surr)")

    @property
    def to_calc_mapping(self):
        """Lookup table from any rendered sentence variant to this side's calc sentence.

        Maps every calc- and pres-convention rendering of the main, surr-x,
        and surr-y sentences (with and without ``convention_mapping``
        applied) to the single corresponding calc-convention target
        sentence — i.e. however a sentence for this relationship side was
        phrased, this dict normalizes it back to the calc form.

        Returns
        -------
        dict[str, str]
        """
        target_main = self.r_calc
        target_surr_rx = self.surr_rx_calc
        target_surr_ry = self.surr_ry_calc
        mapping = {}
        for relation in self._relation_variants(self.var_x, self.var_y):
            mapping[relation] = target_main
        for relation in self._relation_variants(f"{self.var_x} (surr)", self.var_y):
            mapping[relation] = target_surr_rx
        for relation in self._relation_variants(self.var_x, f"{self.var_y} (surr)"):
            mapping[relation] = target_surr_ry
        return mapping

    @property
    def to_pres_mapping(self):
        # Same as to_calc_mapping, but normalizes to the presentation-convention sentence instead.
        target_main = self.r
        target_surr_rx = self.surr_rx
        target_surr_ry = self.surr_ry
        mapping = {}
        for relation in self._relation_variants(self.var_x, self.var_y):
            mapping[relation] = target_main
        for relation in self._relation_variants(f"{self.var_x} (surr)", self.var_y):
            mapping[relation] = target_surr_rx
        for relation in self._relation_variants(self.var_x, f"{self.var_y} (surr)"):
            mapping[relation] = target_surr_ry
        return mapping


class Relationship:
    """Directional x/y relationship, rendered as calc- and presentation-facing sentences.

    A CCM dyad has two directional relationships — "x reconstructs/causes y"
    and "y reconstructs/causes x" — represented here as ``r1`` and ``r2``.
    Each side has both a "calc" sentence (matching the labels actually used
    in output tables, e.g. a ``relation`` column) and a "pres" sentence
    (the human-readable form for figures/text), independently switchable
    between an "operation" phrasing (e.g. "x reconstructs y") and an
    "influence" phrasing (e.g. "y causes x") via ``output_convention``/
    ``pres_convention``. Surrogate variants (``surr_r1x``, ``surr_r2y``,
    etc.) substitute a ``"(surr)"``-suffixed variable name into the same
    sentence patterns. ``to_calc_mapping``/``to_pres_mapping`` build lookup
    tables that normalize any rendered spelling of a relationship back to
    its canonical calc or pres form — used to reconcile relation labels that
    drifted across code versions.

    Parameters
    ----------
    var_x : str, default 'temp'
        Name of the "x" variable.
    var_y : str, default 'TSI'
        Name of the "y" variable.
    surr_flag : {'neither', 'x', 'y', 'both'}, default 'neither'
        Which variable(s) are currently surrogate data, consulted by
        ``set_active_r1``/``set_active_r2``. May also be given as the
        literal value of ``var_x`` or ``var_y``, treated the same as
        ``'x'``/``'y'`` respectively.
    influence_word : str, default 'causes'
        Verb substituted for "causes" in influence-convention sentences.
    operation_word : str, default 'reconstructs'
        Verb substituted for "reconstructs" in operation-convention
        sentences.
    output_convention : {'operation', 'influence'}, default 'influence'
        Sentence form used for each side's calc output.
    pres_convention : {'operation', 'influence'}, default 'influence'
        Sentence form used for each side's presentation output.
    convention_mapping : dict, optional
        Optional word remapping applied when rendering (e.g. to substitute a
        domain-specific verb), passed through to both sides. Default is
        ``{}``.

    Notes
    -----
    ``r1``/``r2`` and their ``surr_*``/``*_calc`` counterparts are all
    properties that delegate to an internal ``RelationshipSide`` instance
    (``self._r1_side``/``self._r2_side``) — see that class for the rendering
    details.

    See Also
    --------
    RelationshipSide : Implements the rendering for one side (r1 or r2).
    ResultsGrid : Consumes a ``Relationship`` to label its half-moon overlay
        markers.
    """

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
        """Build both sides (``r1`` and ``r2``) of an x/y relationship.

        Constructs a ``RelationshipSide`` for each of ``r1`` and ``r2``
        (stored as ``self._r1_side``/``self._r2_side``), passing this
        instance's ``var_x``/``var_y``/word/convention settings through to
        both. All ``r1_*``/``r2_*``/``surr_*`` properties on this class
        delegate to one of those two sides.

        Parameters
        ----------
        var_x : str, optional
            Name of the "x" variable. Default is ``"temp"``.
        var_y : str, optional
            Name of the "y" variable. Default is ``"TSI"``.
        surr_flag : {'neither', 'x', 'y', 'both'}, optional
            Which variable(s) are currently surrogate data, consulted by
            :meth:`set_active_r1`/:meth:`set_active_r2`. May also be set to
            the literal value of ``var_x`` or ``var_y``, treated the same as
            ``'x'``/``'y'`` respectively. Default is ``"neither"``.
        influence_word : str, optional
            Verb substituted for "causes" in pres-convention sentences.
            Default is ``"causes"``.
        operation_word : str, optional
            Verb substituted for "reconstructs" in calc-convention sentences.
            Default is ``"reconstructs"``.
        output_convention : {'operation', 'influence'}, optional
            Sentence form used for each side's calc output. Default is
            ``"influence"``.
        pres_convention : {'operation', 'influence'}, optional
            Sentence form used for each side's presentation output. Default
            is ``"influence"``.
        convention_mapping : dict, optional
            Optional word remapping passed through to both sides. Default is
            ``{}``.
        """
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
        # Mutator: sets self.influence_word and propagates it to both sides. No return value.
        self.influence_word = verb
        self._r1_side.influence_word = verb
        self._r2_side.influence_word = verb

    def set_operation_verb(self, verb):
        # Mutator: sets self.operation_word and propagates it to both sides. No return value.
        self.operation_word = verb
        self._r1_side.operation_word = verb
        self._r2_side.operation_word = verb

    def set_active_r1(self):
        """Return r1's sentence variant matching the current ``surr_flag``.

        Returns
        -------
        str or None
            ``self.surr_r1x``/``self.r1``/``self.surr_r1y``/``self.surr_r1yx``
            depending on whether ``surr_flag`` is ``'x'``-like, ``'neither'``,
            ``'y'``-like, or ``'both'``. Returns ``None`` if ``surr_flag``
            doesn't match any recognized value.
        """
        if self.surr_flag in ("x", self.var_x):
            return self.surr_r1x
        elif self.surr_flag in ("neither",):
            return self.r1
        elif self.surr_flag in ("y", self.var_y):
            return self.surr_r1y
        elif self.surr_flag in ("both",):
            return self.surr_r1yx

    def set_active_r2(self):
        # Same as set_active_r1, but for r2 (returns surr_r2x/r2/surr_r2y/surr_r2yx).
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
        """``r1``'s presentation-facing sentence (delegates to ``self._r1_side.r``).

        Most properties below follow this same delegate-to-``_r1_side``/
        ``_r2_side`` pattern; see :class:`RelationshipSide` for what each
        rendering actually does.

        Returns
        -------
        str
        """
        return self._r1_side.r

    @property
    def r1_calc(self):
        # Calc-convention counterpart of r1.
        return self._r1_side.r_calc

    @property
    def r2(self):
        # r2's presentation-facing sentence (delegates to self._r2_side.r).
        return self._r2_side.r

    @property
    def r2_calc(self):
        # Calc-convention counterpart of r2.
        return self._r2_side.r_calc

    @property
    def surr_r1x(self):
        # r1's sentence with var_x replaced by "var_x (surr)" (presentation convention).
        return self._r1_side.surr_rx

    @property
    def surr_r1x_calc(self):
        # Calc-convention counterpart of surr_r1x.
        return self._r1_side.surr_rx_calc

    @property
    def surr_r1y(self):
        # r1's sentence with var_y replaced by "var_y (surr)" (presentation convention).
        return self._r1_side.surr_ry

    @property
    def surr_r1y_calc(self):
        # Calc-convention counterpart of surr_r1y.
        return self._r1_side.surr_ry_calc

    @property
    def surr_r2x(self):
        # r2's sentence with var_x replaced by "var_x (surr)" (presentation convention).
        return self._r2_side.surr_rx

    @property
    def surr_r2x_calc(self):
        # Calc-convention counterpart of surr_r2x.
        return self._r2_side.surr_rx_calc

    @property
    def surr_r2y(self):
        # r2's sentence with var_y replaced by "var_y (surr)" (presentation convention).
        return self._r2_side.surr_ry

    @property
    def surr_r2y_calc(self):
        # Calc-convention counterpart of surr_r2y.
        return self._r2_side.surr_ry_calc

    @property
    def surr_r2xy(self):
        # Both-surrogate sentence: "x (surr) <influence_word> y (surr)".
        # TODO: review for removal post-Part1A; currently matches surr_r2yx exactly.
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r2yx(self):
        # Both-surrogate sentence: "x (surr) <influence_word> y (surr)".
        # TODO: review for removal post-Part1A; currently matches surr_r2xy exactly.
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r2both(self):
        # Both-surrogate sentence: "x (surr) <influence_word> y (surr)". Returned by set_active_r2 when surr_flag == 'both'.
        return f"{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)"

    @property
    def surr_r1xy(self):
        # Both-surrogate sentence: "y (surr) <influence_word> x (surr)".
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def surr_r1yx(self):
        # Both-surrogate sentence: "y (surr) <influence_word> x (surr)".
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def surr_r1both(self):
        # Both-surrogate sentence: "y (surr) <influence_word> x (surr)". Returned by set_active_r1 when surr_flag == 'both'.
        return f"{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)"

    @property
    def to_calc_mapping(self):
        """Combined r1+r2 lookup table from any rendered sentence to its calc form.

        Union of ``self._r1_side.to_calc_mapping`` and
        ``self._r2_side.to_calc_mapping``.

        Returns
        -------
        dict[str, str]
        """
        mapping = {}
        mapping.update(self._r1_side.to_calc_mapping)
        mapping.update(self._r2_side.to_calc_mapping)
        return mapping

    @property
    def to_pres_mapping(self):
        # Combined r1+r2 counterpart of to_calc_mapping, normalizing to presentation form instead.
        mapping = {}
        mapping.update(self._r1_side.to_pres_mapping)
        mapping.update(self._r2_side.to_pres_mapping)
        return mapping

    def relation_aliases(self, relationship_id):
        """Return all generic, unsuffixed spellings for ``'r1'`` or ``'r2'``.

        This supports category filtering against an output table's ``relation``
        column.  Surrogate-specific spellings belong in ``relation_spec`` and
        are deliberately not included here.
        """
        if relationship_id == "r1":
            return self._r1_side.relation_aliases()
        if relationship_id == "r2":
            return self._r2_side.relation_aliases()
        raise ValueError(f"Unknown relationship category {relationship_id!r}; use 'r1' or 'r2'.")
