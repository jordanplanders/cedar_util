
import logging
logger = logging.getLogger(__name__)
try:
    from cedarkit.utils.cli import log_line
except ImportError:
    from utils.cli.logging import log_line

class RelationshipSide:
    def __init__(self, r, relationship=None, var_x='temp', var_y='TSI', influence_word='causes'):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.var_x = var_x if relationship is None else relationship.var_x
        self.var_y = var_y if relationship is None else relationship.var_y
        self.influence_word = influence_word

        self.surr_rx_count = None
        self.surr_rx_count_outperforming = None
        self.surr_ry_count = None
        self.surr_ry_count_outperforming = None
        self.delta_rho = None
        self.maxlibsize_rho = None
        self.lag = None
        self.surr_rx_outperforming_frac = None
        self.surr_ry_outperforming_frac = None


        # self.surr_rx
        # self.surr_ry

        if r == 'r1':
            self.pattern = 'y causes x'
        elif r == 'r2':
            self.pattern = 'x causes y'


    @property
    def surr_rx(self):
        return self.pattern.replace('x', f'{self.var_x} (surr)').replace('y', self.var_y).replace('causes', self.influence_word)

    @property
    def surr_ry(self):
        return self.pattern.replace('y', f'{self.var_y} (surr)').replace('x', self.var_x).replace('causes', self.influence_word)

    @property
    def r(self):
        return self.pattern.replace('x', self.var_x).replace('y', self.var_y).replace('causes', self.influence_word)



class Relationship:

    def __init__(self, var_x='temp', var_y='TSI', surr_flag='neither'):
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.influence_word = 'causes'
        self.var_x = var_x
        self.var_y = var_y
        self.surr_flag = surr_flag

        # self.active_r1 = self.set_active_r1()
        # self.active_r2 = self.set_active_r2()


    def set_influence_verb(self, verb):
        self.influence_word = verb


    def set_active_r1(self):
        if self.surr_flag in ('x', self.var_x):
            return self.surr_r1x
        elif self.surr_flag in ('neither'):
            return self.r1
        elif self.surr_flag in ('y', self.var_y):
            return self.surr_r1y
        elif self.surr_flag in ('both'):
            return self.surr_r1yx


    def set_active_r2(self):
        if self.surr_flag in ('x', self.var_x):
            return self.surr_r2x
        elif self.surr_flag in ('neither'):
            return self.r2
        elif self.surr_flag in ('y', self.var_y):
            return self.surr_r2y
        elif self.surr_flag in ('both'):
            return self.surr_r2yx

    @property
    def r1(self):
        return f'{self.var_y} {self.influence_word} {self.var_x}'

    @property
    def r2(self):
        return f'{self.var_x} {self.influence_word} {self.var_y}'

    @property
    def surr_r1x(self):
        return f'{self.var_y} {self.influence_word} {self.var_x} (surr)'

    @property
    def surr_r1y(self):
        return f'{self.var_y} (surr) {self.influence_word} {self.var_x}'

    @property
    def surr_r2x(self):
        return f'{self.var_x} (surr) {self.influence_word} {self.var_y}'

    @property
    def surr_r2y(self):
        return f'{self.var_x} {self.influence_word} {self.var_y} (surr)'

    @property
    def surr_r2xy(self):
        return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'

    @property
    def surr_r2yx(self):
        return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'

    @property
    def surr_r2both(self):
        return f'{self.var_x} (surr) {self.influence_word} {self.var_y} (surr)'

    @property
    def surr_r1xy(self):
        return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'

    @property
    def surr_r1yx(self):
        return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'

    @property
    def surr_r1both(self):
        return f'{self.var_y} (surr) {self.influence_word} {self.var_x} (surr)'



