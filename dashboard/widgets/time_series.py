import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes

from dashboard.plot_utils import set_animated
from dashboard.widgets import Widget
from i2mb.utils import global_time


class BaseTimeSeries(Widget):
    def __init__(self, ax: Axes, traces, window_size, num_lines=1, ws_scale="d",
                 sim_length=None, legends=None, title=None):

        sim_length = sim_length is None and 40 * 24 * 12 or sim_length
        self.num_lines = num_lines
        self.ws_scale = ws_scale
        self.traces = traces
        self.window_size = window_size
        self.ax = ax
        self.ax.set_title(title, loc="left")

        self.sim_length = sim_length
        self.__window_size = window_size
        self.wst = global_time.time_scalar * window_size

        self.ax.set_xlabel(f"Time [{ws_scale}]")
        self.ax.set_ylabel("# agents")

        scaler = global_time.time_scalar
        if ws_scale == "Hours":
            scaler = global_time.ticks_hour

        self.ax.set_xticks([t * scaler for t in range(window_size + 1)])
        self.ax.set_xticklabels(["{}".format(v // global_time.time_scalar) for v in ax.get_xticks()])
        self.data = np.ones((sim_length, len(traces)))
        self.data[:, :] = np.nan
        self.data[0, :] = traces
        self.time = np.arange(sim_length)

        if self.num_lines >= 1:
            self.ax.plot(self.time, self.data[:, self.num_lines].T)

        self._draw_content()

        if legends is not None:
            ax.legend(legends, loc="upper left")

        self.ax.set_xlim(0, self.wst * 1.0075)
        ylim_max = 10 > max(traces) * 1.17 and 10 or max(traces) * 1.17
        self.ax.set_ylim(0, ylim_max)

        set_animated(True, self.ax.xaxis)
        set_animated(True, self.ax.yaxis)

    @property
    def window_size(self):
        return self.__window_size

    @window_size.setter
    def window_size(self, v):
        self.__window_size = v
        self.wst = global_time.time_scalar * self.__window_size

    def update_xticks(self, frame):
        if frame > self.wst:
            step = int(frame / (self.__window_size * global_time.time_scalar))
            ul = int(frame / global_time.time_scalar)
            self.ax.set_xticks([r * global_time.time_scalar for r in range(0, ul + 1, step)])
            self.ax.set_xticklabels(["{}".format(
                v // global_time.time_scalar) for v in self.ax.get_xticks()])
            self.ax.set_xlim(0, frame * 1.0075)

    def _draw_content(self):
        pass

    def _update(self, frame):
        return []

    def update(self, frame):
        if frame == len(self.data):
            self.data = np.vstack([self.data, np.full((self.sim_length, len(self.traces)), np.nan)])
            self.time = np.arange(len(self.data))

        artists = [self.ax.xaxis, self.ax.yaxis]
        artists.extend(self._update(frame))
        return artists


class TimeSeriesStacked(BaseTimeSeries):
    def _draw_content(self):
        self.areas = self.ax.stackplot(self.time, self.data[:, self.num_lines:].T,
                                       colors=rcParams["axes.prop_cycle"].by_key()['color'][
                                              self.num_lines:self.data.shape[1]],
                                       alpha=0.7)

    def _update(self, frame):

        self.data[frame, :] = self.traces

        if self.num_lines >= 1:
            for ix, trace, line in zip(range(self.num_lines), self.traces, self.ax.lines):
                self.data[frame, ix] = trace
                line.set_data(self.time, self.data[:, ix])

        self.update_xticks(frame)

        for area in self.areas:
            area.remove()

        self._draw_content()

        ylim_max = 10 > np.nanmax(self.data) * 1.15 and 10 or np.nanmax(self.data) * 1.15
        self.ax.set_ylim(0, ylim_max)

        artists = []
        artists.extend(self.ax.lines)
        artists.extend(self.areas)

        return artists


class TimeSeries(BaseTimeSeries):
    def __init__(self, ax: Axes, traces, window_size, ws_scale="days", sim_length=None, legends=None, title=None):
        super().__init__(ax, traces, window_size, num_lines=0, ws_scale=ws_scale, sim_length=sim_length,
                         legends=legends, title=title)

        self.ax.set_xticklabels([])
        self.ax.set_xlabel(" ")  # we cannot clear the text, otherwise the bbox returns null.

    def _draw_content(self):
        for data, ls, marker, ms in zip(self.data.T,
                                        [" ", " ", " ", " ", " ", " "],
                                        ["X", "o", "*", "+", "x", "3"],
                                        [6, 3, 6, 6, 6, 6]):
            self.ax.plot(self.time, data, alpha=0.9, ls=ls, marker=marker, markersize=ms)

    def _update(self, frame):
        for ix, trace, line in zip(range(len(self.traces)), self.traces, self.ax.lines):
            self.data[frame, ix] = trace
            line.set_data(self.time, self.data[:, ix])

        self.update_xticks(frame)

        ylim_max = np.nanmax(self.data) * 1.15
        if np.isnan(ylim_max) or ylim_max < 10:
            ylim_max = 10

        ylim_min = np.nanmin(self.data) * 1.15
        if np.isnan(ylim_min) or ylim_min > 0:
            ylim_min = 0

        self.ax.set_ylim(ylim_min, ylim_max)

        return self.ax.lines
