import numbers
from textwrap import fill

import numpy as np
from itertools import cycle
from matplotlib import rcParams
from matplotlib.patches import Circle, RegularPolygon, ConnectionPatch
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D


# from https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# From https://stackoverflow.com/questions/24659005/radar-chart-with-multiple-scales-on-multiple-axes
class Radar(object):

    def __init__(self, fig, titles, rect=None):

        self.titles = titles
        if rect is None:
            rect = [0.05, 0.05, 0.95, 0.95]

        self.fig = fig
        self.n = len(titles)
        self.axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i)
                     for i in range(self.n)]

        self.angles = np.arange(0, 360, 360.0 / self.n)
        for ix, ax in enumerate(self.axes):
            if rect[2] < 0.5 or rect[3] < 0.5:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, prune="lower"))

            ax.set_theta_offset(np.deg2rad(self.angles[ix]) + np.pi/2 )
            ax.spines["polar"].set_visible(False)
            # ax.set_rgrids(range(1, 6), angle=self.angles[ix])

        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, [fill(t, 11) for t in titles], wrap=True, )
        for label, angle in zip(self.ax.get_xticklabels(), self.angles):
            if 0.0 < angle < 180:
                label.set_horizontalalignment("right")
            if 180 < angle < 360:
                label.set_horizontalalignment("left")

        for ax in self.axes[:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
            ax.set_rlabel_position(0)

        self.ax.yaxis.set_visible(True)
        self.ax.xaxis.set_visible(True)

    def plot(self, data, *args, **kw):
        color_cycle = self.__get_color_cycle(kw)
        lw_cycle = self.__get_lineweight_cycle(kw)
        ls_cycle = self.__get_linestyle_cycle(kw)
        title = kw.pop("title", None)

        for ix, row in data.loc[:, self.titles].iterrows():
            values = np.r_[row, row[0]]
            pairs = np.vstack([values[:-1], values[1:]]).T
            axis = np.r_[self.axes, self.axes[0]]
            ax_pairs = np.vstack([axis[:-1], axis[1:]]).T
            color = next(color_cycle)
            lw = next(lw_cycle)
            ls = next(ls_cycle)
            for segment, ax_pair in zip(pairs, ax_pairs):
                xyA = 0., segment[0]
                xyB = 0., segment[1]

                con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=ax_pair[0].transData, coordsB=ax_pair[1].transData,
                                      color=color, lw=lw, ls=ls, **kw)

                ax_pair[0].plot([xyA[0]], [xyA[1]])
                ax_pair[1].plot([xyB[0]], [xyB[1]])

                self.fig.add_artist(con)

        for ax in self.axes:
            max_ = np.nanmax(np.array([l2d.get_ydata() for l2d in ax.lines]).flatten())
            ax.set_ylim(0, max_ * 1.15)

        for ax in self.axes[1:]:
            ax.yaxis.grid(False)
            ax.tick_params(axis='y', pad=0, left=True, length=6, width=0.6, direction='out', color="gray")

        update_title = self.ax.get_title() == ""
        if title is not None:
            self.ax.set_title(title)
            self.ax.title.set(fontweight="bold")
            bbox = self.ax.get_position()
            title_bbox = self.ax.title.get_window_extent(self.fig.canvas.get_renderer())
            title_height = title_bbox.height / self.fig.get_window_extent().height
            title_offset = 1 - self.ax.title.get_position()[1]
            if update_title:
                for ax in self.axes:
                    ax.set_position([bbox.p0[0], bbox.p0[1], bbox.width, bbox.height - title_height - title_offset])

    @staticmethod
    def __get_prop_cycle(prop_name, kw, alt_prop_name=None, default_value=None):
        prop = kw.pop(prop_name, None)
        if prop is None and alt_prop_name is not None:
            prop = kw.pop(alt_prop_name, None)

        if prop is None:
            try:
                prop = rcParams["axes.prop_cycle"].by_key()[prop_name]
            except KeyError:
                prop = default_value

        prop_cycle = cycle([prop])
        if not (isinstance(prop, str) or isinstance(prop, numbers.Number) or prop is None):
            prop_cycle = cycle(prop)

        return prop_cycle

    def __get_color_cycle(self, kw):
        return self.__get_prop_cycle("color", kw)

    def __get_lineweight_cycle(self, kw):
        return self.__get_prop_cycle("lw", kw, "linewidth", 1.2)

    def __get_linestyle_cycle(self, kw):
        return self.__get_prop_cycle("ls", kw, "linestyle")

    def get_r_lims(self):
        y_lims = []
        for ax in self.axes:
            y_lims.append(ax.get_ylim())

        return np.array(y_lims)

    def set_r_lims(self, lims):
        for ax, lim in zip(self.axes, lims):
            ax.set_ylim(0, lim)
