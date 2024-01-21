from rapp.commands import analysis
from rapp.commands import optical_rotation
from rapp.commands import phase_diff
from rapp.commands import plot_raw
from rapp.commands import polarimeter
from rapp.commands import simulations

COMMANDS = {
    'analysis': analysis,
    'OR': optical_rotation,
    'phase_diff': phase_diff,
    'plot_raw': plot_raw,
    'polarimeter': polarimeter,
    'sim': simulations,
}
