"""ITMO-inspired presentation colors; no effect on analysis.

Brand HEX values: https://itmo.ru/file/form_style/brandbook.pdf, page 30.
Neutrals and dark cyan are UI/print contrast variants.
"""

BLUE = "#0B68FE"
CYAN = "#00CCFF"
VIOLET = "#7F00FF"
RED = "#F93F37"
LIME = "#D0FF14"
CYAN_DARK = "#0084A6"
INK = "#111111"
MUTED = "#5B5E66"
LINE = "#DDE1EB"
SURFACE = "#F3F6FC"
WHITE = "#FFFFFF"

PLOT_STYLE = {
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LINE,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
}


def decimal_comma(fig):
    """Use decimal commas in numeric tick labels of all axes of a matplotlib figure."""
    from matplotlib.ticker import ScalarFormatter

    class CommaFormatter(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(CommaFormatter())
