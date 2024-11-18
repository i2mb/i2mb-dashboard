import matplotlib.text as text
import seaborn as sb


# From https://stackoverflow.com/questions/38463369/subtitles-within-matplotlib-legend
from matplotlib import rcParams
from matplotlib.artist import Artist


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = text.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def create_radar_plot_grid(nrows=1, ncols=1, top_padding=0, bottom_padding=0, left_padding=0, right_padding=0,
                           v_space=0, h_space=0):

    available_width = 1 - left_padding - right_padding
    available_height = 1 - top_padding - bottom_padding
    box_width = (available_width - (ncols - 1) * h_space) / ncols
    box_height = (available_height - (nrows - 1) * v_space) / nrows
    for row in reversed(range(nrows)):
        for column in range(ncols):
            x_pos = left_padding + column * box_width + column * h_space
            y_pos = bottom_padding + row * box_height + row * v_space
            yield x_pos, y_pos, box_width, box_height


def fit_axes_to_bbox(ax, base_left, base_bottom, base_right, base_top, adjust_legend=None):
    fig = ax.get_figure()
    left, bottom, right, top = [rcParams[param] for param in ["figure.subplot.left", "figure.subplot.bottom",
                                                              "figure.subplot.right", "figure.subplot.top"]]
    fl, fb, fr, ft = fig.get_window_extent().extents
    [left, bottom], [right, top] = fig.transFigure.transform([[left, bottom], [right, top]])

    # Tight Boxes
    renderer = fig.canvas.get_renderer()

    ll, lb, lw, lh = get_tight_box_bounds(ax.get_legend(), renderer)

    yall, yalb, yalw, yalh = get_tight_box_bounds(ax.yaxis.label, renderer)
    xall, xalb, xalw, xalh = get_tight_box_bounds(ax.xaxis.label, renderer)

    ytl, ytb, ytw, yth = get_tight_box_bounds(ax.yaxis, renderer)
    xtl, xtb, xtw, xth = get_tight_box_bounds(ax.xaxis, renderer)

    # Axes position
    al = fr * base_left + left + yalw + ytw
    ab = ft * base_bottom + xalh + xth + bottom
    right_pad = fr - right
    aw = fr * base_right - right_pad - al
    ah = ft * base_top - ab
    if adjust_legend == "top":
        ah -= lh

    [al_, ab_], [aw_, ah_], = fig.transFigure.inverted().transform([[al, ab], [aw, ah]])
    ax.set_position([al_, ab_, aw_, ah_])

    if adjust_legend == "top":
        # Legend position
        ll = fr * base_left + left + yalw
        lb = ah + ab + 8.
        lw = fr * base_right - right_pad - ll
        lh = lh
        [ll, lb], [lw, lh] = fig.transFigure.inverted().transform([[ll, lb], [lw, lh]])

        sb.move_legend(ax, "upper left", title_fontsize="small", fontsize="small", ncol=3,
                       mode="expand",
                       bbox_to_anchor=[ll, lb, lw, lh],
                       borderaxespad=0.0,
                       bbox_transform=fig.transFigure)


def get_tight_box_bounds(artist, renderer):
    left = bottom = width = height = 0
    tb = artist.get_tightbbox(renderer)
    if tb is not None:
        left, bottom, width, height = tb.bounds

    return left, bottom, width, height


def overline(value):
    return f"$\\mathrm{{\\overline{value}}}$"


def set_animated(b: bool, artist: Artist):
    artist.set_animated(b)
    for child in artist.get_children():
        set_animated(b, child)


# Brought from future matplotlib
def _val_or_rc(val, rc_name):
    """
    If *val* is None, return ``mpl.rcParams[rc_name]``, otherwise return val.
    """
    return val if val is not None else rcParams[rc_name]