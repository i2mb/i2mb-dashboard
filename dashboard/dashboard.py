#  dct_mct_analysis
#  Copyright (C) 2021  FAU - RKI
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, cm, cycler
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.text import Text

from dashboard.color_utils import rgba_to_rgb
from dashboard.widgets import Widget
from dashboard.widgets.text import StaticText, Table, SimClock, DynamicText
from dashboard.widgets.time_series import TimeSeriesStacked, TimeSeries
from i2mb.pathogen import UserStates
from i2mb.utils import global_time
from i2mb.utils.visualization.scale_bar import scale_bar

# from simulator import logos

CM = rcParams["axes.prop_cycle"].by_key()['color']
CM[6] = CM[2]
CM[2] = "k"
CM[5] = CM[4]  # Do not distinguish between pre and post symtomatic.
bg_cache = {}

cmap = list(cm.get_cmap("tab10").colors)
cmap[5] = '#ffd92f'
rcParams['axes.prop_cycle'] = cycler(color=cmap)


# rcParams['axes.prop_cycle'] = cycler(color=cm.get_cmap("viridis")(np.linspace(0., 1., 7)))


class PathogenStateTimeSeries(Widget):
    ORDER = [UserStates.infectious, UserStates.infected, UserStates.exposed, UserStates.susceptible,
             UserStates.immune, UserStates.deceased]

    def __init__(self, ax: Axes, population, pathogen, window_size, ws_scale="days", sim_length=None):
        self.sim_length = sim_length is None and 40 * 12 * 24 or sim_length
        self.__window_size = window_size
        self.pathogen = pathogen
        self.wst = global_time.time_scalar * window_size
        self.ax = ax
        self.ax.set_title("Agent States", loc="left")

        self.ax.set_xlabel(f" ")
        self.ax.set_ylabel("# agents")

        custom_scat = [plt.Rectangle([0, 0], 1, 1, color=rgba_to_rgb(CM[i], alpha=0.7))
                       for i, l in enumerate(SpaceMap.LEGEND_LABELS)]

        scaler = global_time.time_scalar
        if ws_scale == "Hours":
            scaler = global_time.ticks_hour

        self.ax.set_xticks([t * scaler for t in range(window_size + 1)])
        self.ax.set_xticklabels(["{}".format(v // global_time.time_scalar) for v in ax.get_xticks()])

        self.data = np.zeros((len(UserStates), self.sim_length), dtype=int)
        # self.data[PathogenStateTimeSeries.ORDER.index(UserStates.susceptible), :] = len(population)
        self.time = np.arange(self.sim_length)
        self.areas = self.ax.stackplot(self.time, *self.data, colors=[CM[u] for u in PathogenStateTimeSeries.ORDER],
                                       alpha=0.8)

        self.ax.legend([custom_scat[i] for i in SpaceMap.ORDER],
                       [SpaceMap.LEGEND_LABELS[i] for i in SpaceMap.ORDER],
                       loc="upper left")

        self.ax.set_xlim(0, self.wst * 1.015)
        self.ax.set_xticklabels([])
        self.ax.set_ylim(0, len(population))
        self.ax.xaxis.set_animated(True)
        # self.base_children = self.ax.get_children()

    @property
    def window_size(self):
        return self.__window_size

    @window_size.setter
    def window_size(self, v):
        self.__window_size = v
        self.wst = global_time.time_scalar * self.__window_size

    def update(self, frame):
        if frame == self.data.shape[1]:
            self.data = np.hstack([self.data, np.zeros((len(UserStates), self.sim_length), dtype=int)])
            self.time = np.arange(self.data.shape[1])

        points = self.pathogen.get_totals()
        # self.time.append(frame)
        for d, t in zip(self.data, PathogenStateTimeSeries.ORDER):
            d[frame] = points[t]

        # self.ax._children = self.base_children
        if frame > self.wst:
            step = int(frame / (self.__window_size * global_time.time_scalar))
            ul = int(frame / global_time.time_scalar)
            self.ax.set_xticks([r * global_time.time_scalar for r in range(0, ul + 1, step)])
            # self.ax.set_xticklabels(["{}".format(
            #     v // global_time.time_scalar) for v in self.ax.get_xticks()])
            self.ax.set_xlim(0, frame * 1.0075)
            # self.base_children = self.ax.get_children()

        for area in self.areas:
            area.remove()

        self.areas = self.ax.stackplot(self.time, *self.data, colors=[CM[u] for u in PathogenStateTimeSeries.ORDER],
                                       alpha=0.7, zorder=-1)

        return self.areas + [self.ax.get_legend(), self.ax.xaxis]


class SpaceMap(Widget):
    LEGEND_LABELS = [u.name.title() for u in UserStates if u != UserStates.infectious]
    LEGEND_LABELS[UserStates.exposed] = "Exposed"
    LEGEND_LABELS[UserStates.infected] = "Infected"
    ORDER = [UserStates.susceptible, UserStates.exposed, UserStates.infected,
             UserStates.immune, UserStates.deceased]

    def __init__(self, ax: Axis, scenario, population):
        self.scenario = scenario
        self.population = population
        self.ax, self.scat = scenario.draw_world(ax=ax, padding=0.01, border=True)
        self.scale_bar = scale_bar(self.ax, position=(0.9045, 0.08), )

        # self.ax.set_position([0.01, 1. - h])
        self.legend()

    def init(self):
        return [self.scat]

    def update(self, frame):
        state = self.population.state.copy()
        self.scat.set_offsets(self.scenario.relocator.get_absolute_positions())
        self.scat.set_color([CM[s] for s in state.ravel().astype(int)])
        return self.scat,

    def legend(self):
        custom_scat = [plt.Line2D([], [], color='w', lw=0) for l in SpaceMap.LEGEND_LABELS]
        for i, r in enumerate(SpaceMap.LEGEND_LABELS):
            l1 = custom_scat[i]
            l1.set_marker('o')
            l1.set_markersize(10)
            # l1.set_markevery(-10)
            l1.set_markeredgecolor(CM[i])
            l1.set_alpha(0.7)
            l1.set_markerfacecolor(CM[i])

        self.ax.legend([custom_scat[i] for i in SpaceMap.ORDER], [SpaceMap.LEGEND_LABELS[i] for i in SpaceMap.ORDER],
                       bbox_to_anchor=(0, 1, 1, 1),
                       ncols=len(SpaceMap.ORDER),
                       mode="expand",
                       loc='lower left')


class LogoSpecs:
    def __init__(self, path, ax=None, scale=None, xy=None):
        self.ax = ax
        self.path = path
        self.xy = xy
        self.scale = scale


class Background(Widget):
    def __init__(self, logos: dict[str, LogoSpecs]):
        self.logo_specs = logos
        for logo in self.logo_specs.values():
            if logo.ax is None:
                continue

            Background.show_logo(logo.path, logo.ax, logo.xy, logo.scale)

    @staticmethod
    def show_logo(f_name, logo_ax, xy=None, scaling=1., alpha=1.):
        if xy is None:
            xy = 0, 0, 0., 0.1

        if scaling is None:
            scaling = 1.

        logo_image = plt.imread(f_name)
        logo_ax.imshow(logo_image, alpha=alpha, animated=False)
        logo_ax.set_axis_off()
        # logo_ax.yaxis.set_ticks([])
        # logo_ax.xaxis.set_ticks([])


class StatBars(Widget):
    locations_ = ["home", "office", "bar", "restaurant", 'bus', 'car']

    def __init__(self, ax: Axes, population, title=None):
        self.population = population
        self.ax = ax
        self.bars = self.ax.barh([l.title() for l in StatBars.locations_], [0] * len(StatBars.locations_))
        self.ax.set_xlim(0, len(population) * .315)
        self.ax.set_title(title, loc="left")
        self.ax.set_xlabel("# agents")
        self.ax.xaxis.set_animated(True)

        self.labels = [self.ax.text(0, i, "", va="center", color='black', fontweight="bold") for i, l in
                       enumerate(StatBars.locations_)]

    def update(self, frame):
        if hasattr(self.population, "location_contracted"):
            cpl = self.population.location_contracted == DynamicText.locations_
        else:
            cpl = np.zeros((len(self.population), 1))

        cpl = cpl.sum(axis=0)
        label: Text
        for bar, value, label in zip(self.bars, cpl, self.labels):
            bar.set_width(value)
            if value > 0:
                label.set_text(f" {value / cpl.sum() * 100:.0f}% ")
                width, height = abs(self.ax.transData.inverted().transform(label.get_tightbbox().size))
                label_position = (value, bar.get_bbox().height / 2 + bar.get_bbox().y0)
                label.set_position(label_position)
                if width < value:
                    label.set_horizontalalignment("right")
                    label.set_color("white")
                else:
                    label.set_horizontalalignment("left")
                    label.set_color("black")

            else:
                label.set_text("")
            # if label.get_bbox_patch().w < value:
            #     x, y = label.get_position()
            #     label.set_position(x, y)

            if value > 30:
                self.ax.set_xlim(0, 60*1.15)
            elif value > 60:
                self.ax.set_xlim(0, 100*1.15)

        artists = list(self.labels)
        artists.extend(self.bars)
        artists.append(self.ax.xaxis)
        return artists


class Dashboard:
    def __init__(self, scenario, population, pathogen, fig_size_inches=(15, 8),
                 intervention=None,
                 show_traces=True, traces=None, legends=None, use_state=False,
                 experiment_properties=None, window_size=8, sim_length=1000000,
                 show_stacked=True, stacked_traces=None, stacked_legend=None, stacked_title=None,
                 time_series_title=None, logos=None):

        if logos is None:
            logos = {}

        num_plots = 3 + len(logos) + 3

        self.figure = plt.figure(figsize=fig_size_inches, constrained_layout=True)
        main_spec = plt.GridSpec(nrows=2, ncols=1, figure=self.figure, height_ratios=[1, 0.1])
        logos_specs = main_spec[1].subgridspec(nrows=1, ncols=len(logos))
        dashboard_area = main_spec[0].subgridspec(nrows=1, ncols=2)
        plt_area = dashboard_area[1].subgridspec(nrows=3, ncols=1, height_ratios=[0.9, 0.9, 0.9], )

        map_area = dashboard_area[0].subgridspec(nrows=3, ncols=1, height_ratios=[0.25, 1, 0.75])
        static_text_area = map_area[0].subgridspec(nrows=1, ncols=2)
        dynamic_text_area = map_area[2].subgridspec(nrows=1, ncols=2)

        static_text_ax = self.figure.add_subplot(static_text_area[0, 1])
        sim_clock_ax = self.figure.add_subplot(static_text_area[0, 0])

        space_map = self.figure.add_subplot(map_area[1, 0])

        bar_ax = self.figure.add_subplot(dynamic_text_area[0, 0])
        dynamic_text_ax = self.figure.add_subplot(dynamic_text_area[0, 1])
        logo_order = ["fau", "cdh", "hs", "ufr", "ies", "rki"]

        for l_spec, logo in zip(logos_specs, logo_order):
            logos[logo].ax = self.figure.add_subplot(l_spec)

        stacked = self.figure.add_subplot(plt_area[2, 0])
        pathogen_states = self.figure.add_subplot(plt_area[0, 0])
        time_series = self.figure.add_subplot(plt_area[1, 0])

        self.space_map = SpaceMap(space_map, scenario, population)
        self.static_text_ax = StaticText(static_text_ax, population, experiment_properties["sim_engine"],
                                         intervention=intervention, title="Run Parameters")
        self.sim_clock_ax = SimClock(sim_clock_ax, population, experiment_properties["sim_engine"],
                                     intervention=intervention)

        self.bars = StatBars(bar_ax, population, title="Location of infection")
        self.dynamic_text_ax = Table(dynamic_text_ax, [
            ["Clearance period:", "Incubation period:", "Generation interval:", "Serial interval:"],
            ["7-day hosp. incidence:", "7-day incidence:", "Affected population:"],
            ["Avg. Days in quarantine:", "Avg. Days in isolation:", "Avg. confinements:",
             "False positive rate:", "False negative rate:", "Currently confined:"]],
                                     section_titles=["     Pathogen:", "     Population:", "     Agent:"],
                                     titles_properties={"ha": "left"},
                                     title="Pandemic characteristics",
                                     row_value_formats=[
                                             " {0:.1f} d",  # Avg. Clearance period
                                             " {0:.1f} d",  # Avg. Incubation period
                                             " {0:.1f} d",  # Avg. Generation interval
                                             " {0:.1f} d",  # Avg. Serial interval
                                             " {0:.1f} d",  # 7-day hosp. incidence
                                             " {0:.1f} d",  # 7-day incidence
                                             " {0:d}",  # Affected population
                                             " {0:.1f} d",  # Avg. Days in quarantine
                                             " {0:.1f} d",  # Avg. Days in isolation
                                             " {0:.1f}",  # Avg. confinements
                                             " {0:.1f}%",  # False positive rate
                                             " {0:.1f}%",  # False negative rate
                                             " {0:d}"  # Currently confined
                                         ]
                                    )
        self.background = Background(logos)

        self.pathogen_states = PathogenStateTimeSeries(pathogen_states, population, pathogen, window_size=window_size,
                                                       sim_length=sim_length)
        self.time_series = TimeSeries(time_series, traces, window_size=window_size, sim_length=sim_length,
                                      legends=legends, title=time_series_title)
        self.stacked = TimeSeriesStacked(stacked, stacked_traces, window_size=window_size, sim_length=sim_length,
                                         legends=stacked_legend, title=stacked_title)

        # self.arrange_axes()
        self.__widgets = [self.space_map, self.sim_clock_ax, self.bars,
                          self.static_text_ax,
                          self.dynamic_text_ax,
                          self.pathogen_states,
                          self.time_series,
                          self.stacked]

    def init(self):
        artist = []
        for widget in self.__widgets:
            artist.extend(widget.init())

        return artist

    def update(self, frame):
        artist = []
        for widget in self.__widgets:
            artist.extend(widget.update(frame))

        return artist



