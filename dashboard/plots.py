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
from numbers import Number
from operator import itemgetter

import ipywidgets as widgets
import numpy as np
import pandas as pd
import seaborn as sb

from IPython.core.display import display
from ipywidgets import Output
from itertools import cycle, product
from matplotlib import rcParams, cm, pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from seaborn import despine

from dashboard.parallel_coordinates import ParallelCoordinates
from dashboard.spider_plot import Radar
from i2mb.utils import global_time


def min2timestr(mins, pos=0):
    # Match TicksFormatter API
    hour = int(mins // 60)
    minutes = int(mins % 60)
    return f"{hour:02}:{minutes:02}"


def truncated_distplot(data, dist_limits, axlabel=None, label=None, hist=False, bins=False, ax=None, **kwargs):
    if len(data.shape) == 1:
        if type(data) is pd.Series:
            data = data.values

        data = data.reshape(-1, 1)

    if len(data.shape) > 2:
        raise RuntimeError("Expecting data to have at most 2D")

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    left_boundary, right_boundary = dist_limits
    if left_boundary is None:
        left_boundary = -np.inf

    if right_boundary is None:
        right_boundary = np.inf

    data_long = data.ravel()
    data_long = data_long[~np.isnan(data_long)]
    data_long = data_long[~np.isinf(data_long)]
    _range = max(data_long) - min(data_long)
    _type = 0  # Truncated on both sides
    if left_boundary == -np.inf:
        # We assume non-truncated distribution were handled else where
        left_boundary = min(data_long) - np.std(data_long)
        _type = 1  # Truncated on the right

    if right_boundary == np.inf:
        right_boundary = max(data_long) + np.std(data_long)
        _type = -1  # Truncated on the left

    # 100 points is usually good enough for plotting.
    xlim = kwargs.get("xlim", None)
    if xlim is None:
        xlim = left_boundary - _range * 0.15, right_boundary + _range * 0.15

    x = np.arange(*xlim, _range * 1.3 / 100)

    left_pad = np.abs(x - left_boundary).argmin()
    right_pad = np.abs(x - right_boundary).argmin()
    if _type == 1:
        left_pad = 0

    if _type == 1:
        right_pad = len(x)

    for column in range(data.shape[-1]):
        vals = data[:, column][~np.isnan(data[:, column])]
        vals = vals[~np.isinf(vals)]
        kde = gaussian_kde(vals)
        truncated_area = kde.integrate_box(left_boundary, right_boundary)
        pdf = kde.pdf(x) / truncated_area
        pdf[:left_pad] = 0
        pdf[right_pad:] = 0
        ax.plot(x, pdf, label=label, **kwargs)

    ax.set_xlabel(axlabel)

    return ax


def draw_daily_distribution(x, reference_lines=None, truncate=None, hist=False, **kwargs):
    """
    x is assumed to be in unit days
    Parameters
    ----------
    x
    reference_lines
    truncate
    hist

    Returns
    -------
    ax
    """
    if reference_lines is None:
        reference_lines = []

    axlabel = kwargs.pop("axlabel", "[Days]")
    label = kwargs.pop("label", "I2MB")
    ax = kwargs.pop("ax", None)
    xlim = kwargs.pop("xlim", None)
    legend = kwargs.pop("legend", True)
    if not list(x) == []:
        bins = kwargs.pop("bins", np.arange(int(np.nanmin(x)) - 1, int(np.nanmax(x)) + 1))  # TODO: Plot Histogram

        if truncate is None:
            kwargs.update(kwargs.pop("kde_kws", {}))
            ax = sb.kdeplot(x, label=label, ax=ax, **kwargs)
            ax.set_xlabel(axlabel)
        else:
            line_kws = kwargs.pop("kde_kws", {})
            ax = truncated_distplot(x, dist_limits=truncate, axlabel=axlabel, label=label, bins=bins, ax=ax, **line_kws,
                                    **kwargs)

    for r_l in reference_lines:
        ax.plot(r_l["t"], r_l["y"], label=r_l.get("label"))

    ax.set_ylabel("PDF")

    if xlim is not None:
        ax.set_xlim(xlim)

    if not legend:
        ax.legend().remove()
    else:
        ax.legend(fontsize="small", bbox_to_anchor=(0.8, 1.1))

    return ax


def draw_distributions_per_experiment(data_frame, refs=None, axlabel="Days", **kwargs):
    if refs is None:
        refs = []

    rcParams["font.size"] = 18
    plt.close("all")
    grouping_level = 0
    outputs = []
    output = widgets.Output()
    outputs.append(output)
    legend = kwargs.pop("legend", True)
    with output:
        fig, ax = plt.subplots(figsize=(8, 4))
        # ax = None
        ax = fill_distribution_per_experiment_axis(data_frame, ax, axlabel, refs, **kwargs)
        if legend:
            ax.legend(fontsize=12, bbox_to_anchor=(0.8, 1.1), loc=2)
        else:
            ax.legend().remove()

        fig.tight_layout()
        display(fig)
        plt.close("all")

    return outputs


def fill_distribution_per_experiment_axis(data_frame, ax, axlabel, refs=None, **kwargs):
    legend = kwargs.pop("legend", True)
    for k, data in data_frame.items():
        ax = draw_daily_distribution(data, ax=ax, label=k, axlabel=axlabel, legend=legend, hist=False, **kwargs)

    draw_daily_distribution([], ax=ax, legend=legend, reference_lines=refs)
    return ax


def draw_individual_distribution_per_experiment(data_frame, refs=None, axlabel="Days"):
    plt.close("all")
    if refs is None:
        refs = []

    outputs = []
    for k, data in data_frame.groupby(level=0, axis=1):
        output = widgets.Output()
        outputs.append(output)
        with output:
            fig = plt.figure()
            ax = None
            for col_name, col_data in data.items():
                ax = draw_daily_distribution(col_data, ax=ax, label=None, hist=False, color="#e6e6e6",
                                             axlabel=axlabel)

            draw_daily_distribution([], ax=ax, reference_lines=refs)
            ax.set_title(k)
            ax.legend(fontsize=12)
            fig.tight_layout()
            display(fig)
            plt.close("all")

    return outputs


def draw_time_series_sb(data_frame, ax_xlabel="Days", ax_ylabel=None, **kwargs):
    """Draws time series data using seaborn lineplot. It is very slow"""
    legend = kwargs.pop("legend", False)
    ax = sb.lineplot(data=data_frame, legend=False, **kwargs)

    if legend:
        ax.legend(fontsize=12, bbox_to_anchor=(0.8, 1.1))

    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel(ax_ylabel)
    ax.figure.tight_layout()

    return ax.get_figure(), ax


def draw_time_series(data_frame, ax_xlabel="Days", ax_ylabel=None, ax=None, ci=None, **kwargs):
    """Draws time series data using mean and 5% confidence interval."""
    mean_ = data_frame.groupby(level=0, axis=1).mean()

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    mean_.plot(ax=ax, **kwargs)

    if ci is not None:
        ci_95 = data_frame.groupby(level=0).std() * ci
        ub = mean_ + ci_95
        lb = mean_ - ci_95

        alpha = 0.3 * kwargs.pop("alpha", 1)
        ax.fill_between(ub.index, ub, lb, alpha=alpha, **kwargs)

    ax.set_xlabel(ax_xlabel)
    ax.set_ylabel(ax_ylabel)

    legend = kwargs.pop("legend", False)
    if legend:
        ax.legend(fontsize=12, bbox_to_anchor=(0.5, 1.1), loc=2)

    return fig, ax


def draw_advanced_axes_plot(kind, data, fig, references=None, grid_spec=None, variable_names=None,
                            refs_kwargs=None,
                            advanced_ax_kwargs=None, advanced_ax=None,
                            maximums=None):
    if refs_kwargs is None:
        refs_kwargs = {}

    if advanced_ax_kwargs is None:
        advanced_ax_kwargs = {}

    if variable_names is None:
        try:
            variable_names = data.columns
        except AttributeError:
            if len(variable_names.shape) == 1:
                variable_names = [f"var_{i}" for i in range(variable_names.shape[0])]
            else:
                variable_names = [f"var_{i}" for i in range(variable_names.shape[1])]

    sharedx = advanced_ax_kwargs.pop("sharedx", False)
    rot = advanced_ax_kwargs.pop("rot", False)
    if advanced_ax is None:
        if kind == "pc":
            advanced_ax = ParallelCoordinates(fig, variable_names, grid_spec=grid_spec, rot=rot, sharedx=sharedx)
        elif kind == "spider":
            advanced_ax = Radar(fig, variable_names, rect=grid_spec)
        else:
            raise TypeError(f"kind must be one of ['pc', 'spider'], but {kind} provided.")

    if references is not None:
        if type(references) is list:
            if type(refs_kwargs) is not list:
                refs_kwargs = [refs_kwargs] * len(references)

            for ref, ref_prop in zip(references, refs_kwargs):
                __plot_data(advanced_ax, ref.loc[:, variable_names], ref_prop.copy(), use_title=False)
        else:
            __plot_data(advanced_ax, references.loc[:, variable_names], refs_kargs.copy(), use_title=False)

    __plot_data(advanced_ax, data.loc[:, variable_names], advanced_ax_kwargs.copy(), selected_only=False)
    fill = advanced_ax_kwargs.pop("fill", False)
    if kind == "pc" and fill:
        fill_between_min_max(advanced_ax, advanced_ax_kwargs, data, variable_names)

    if maximums is not None:
        advanced_ax.set_r_lims(maximums)

    return advanced_ax


def fill_between_min_max(advanced_ax, advanced_ax_kwargs, data, variable_names):
    path1 = data.loc[:, variable_names].max()
    path2 = data.loc[:, variable_names].min()
    color, lw, _, _ = __prepare_parameters(advanced_ax_kwargs, data, False)

    advanced_ax.fill_between(path1, path2, lw=list(lw), color=list(color), **advanced_ax_kwargs.copy())


def __plot_data(advanced_axis, data, advanced_ax_kwargs, selected_only=True, use_title=True):
    color, lw, selected, selected_mask = __prepare_parameters(advanced_ax_kwargs, data, use_title)

    if selected is not None and selected_only:
        lw = lw[selected_mask]
        color = color[selected_mask]
        if type(selected) is str:
            data = data.loc[[selected]]
        else:
            data = data.loc[selected]

    advanced_axis.plot(data, lw=list(lw), color=list(color), **advanced_ax_kwargs)


def __prepare_parameters(advanced_ax_kwargs, data, use_title):
    if not use_title:
        advanced_ax_kwargs.pop("title", None)
    selected = advanced_ax_kwargs.pop("selected", None)
    selected_mask = parse_selected_mask(data, selected)
    lw = parse_line_weight(data, advanced_ax_kwargs, selected_mask)
    color = parse_color(data, advanced_ax_kwargs)
    return color, lw, selected, selected_mask


def parse_selected_mask(data, selected):
    if selected is not None:
        if type(data.index) is pd.MultiIndex:
            index_hypercube = np.array([list(i) for i in data.index.values]).reshape(1, 1, -1, 4)
            selected_hypercube = np.array([list(i) for i in selected]).reshape(-1, 4, 1, 1)
            selected_mask = (index_hypercube == selected_hypercube).any(axis=1).all(axis=2).any(axis=0)
        else:
            selected_mask = (data.index.values.reshape(-1, 1) == selected).any(axis=1)
    else:
        selected_mask = np.zeros(len(data), dtype=bool)
    return selected_mask


def parse_line_weight(data, radar_kwargs, selected_mask):
    lw = radar_kwargs.pop("lw", rcParams["axes.linewidth"])
    if isinstance(lw, Number):
        lw = [lw] * len(data)

    lw = np.array(lw)
    selected_lw = radar_kwargs.pop("selected_lw", rcParams["axes.linewidth"])
    lw[selected_mask] = selected_lw
    return lw


def parse_color(data, radar_kwargs):
    color = radar_kwargs.pop("color", None)

    if color is None:
        color = [next(iter(rcParams["axes.prop_cycle"]))["color"]]

    if len(color) == 1:
        color = color * len(data)

    color = np.array(color)

    cm = radar_kwargs.pop("cm", None)
    if cm is not None:
        if isinstance(cm, Colormap):
            color = np.array([cm(i) for i in np.linspace(0, 1, len(data))])
        else:
            color = np.array(cm)

    return color


def activity_timeseries(id_, data, time_factor_ticks_day=None, axs=None, hbar_kwargs=None, wrap=True):
    if hbar_kwargs is None:
        hbar_kwargs = {}

    color = hbar_kwargs.get("color")
    activity_color = {}
    if color is None:
        activity_color_cycler = cycle(cm.get_cmap("tab20c").colors)
        # activity_color_cycler = cycle(cm.get_cmap("tab20").colors)
        activity_color = dict(data.activity.cat.categories.map(lambda x: (x, next(activity_color_cycler)), "ignore"))
        color = data["activity"].astype(str).apply(lambda x: activity_color[x]).values.ravel()
    else:
        if isinstance(color, dict):
            activity_color = color
            color = data["activity"].astype(str).apply(lambda x: activity_color[x]).values.ravel()

    if time_factor_ticks_day is None:
        time_factor_ticks_day = global_time.ticks_day

    minutes_per_ticks = (60 * 24) / time_factor_ticks_day

    if isinstance(id_, int):
        id_ = [id_]

    if axs is None:
        fig, axs = plt.subplots(figsize=(14.16, 8.91), nrows=len(id_), sharex=True, squeeze=False)
        axs = axs.ravel()
    else:
        fig = axs[0].get_figure()

    for _id_, ax in zip(id_, axs):
        id_data = data.set_index("id").loc[_id_, :]
        if wrap:
            wrap_selector = id_data.start + id_data.duration > (id_data.day + 1) * time_factor_ticks_day
            wrapped_activities = id_data[wrap_selector].copy()
            wrapped_activities.loc[:, "day"] += 1
            wrapped_activities.loc[:, "duration"] -= time_factor_ticks_day - (
                    wrapped_activities["start"] % time_factor_ticks_day)
            id_data.loc[wrap_selector, "duration"] = time_factor_ticks_day - (
                    wrapped_activities["start"] % time_factor_ticks_day)
            wrapped_activities.loc[:, "start"] = 0

        color_ = color[data["id"] == _id_]

        # parallel_starts = (id_data.loc[:, ["start"]].values.ravel() -
        #                    (id_data.loc[:, ["day"]].values.ravel() * time_factor_ticks_day)).astype(float)
        parallel_starts = (id_data.loc[:, ["start"]].values.ravel() % time_factor_ticks_day).astype(float)
        parallel_starts *= minutes_per_ticks
        ax.barh(id_data.loc[:, ["day"]].values.ravel() * -1,
                id_data.loc[:, ["duration"]].values.ravel() * minutes_per_ticks,
                0.8,
                parallel_starts,
                color=color_,
                **hbar_kwargs)

        if wrap:
            ax.barh(wrapped_activities.loc[:, ["day"]].values.ravel() * -1,
                    wrapped_activities.loc[:, ["duration"]].values.ravel() * minutes_per_ticks,
                    0.8,
                    wrapped_activities.loc[:, ["start"]].values.ravel(),
                    color=color_[wrap_selector],
                    **hbar_kwargs)

        ax.set_ylabel(f"Id {_id_} - Day")

        ax.yaxis.set_major_formatter(lambda x, pos: int(x * -1))
        ax.xaxis.set_major_formatter(min2timestr)

    patches = [Rectangle([0, 0], 0, 0, color=c) for c in activity_color.values()]
    axs[-1].legend(patches, activity_color, loc="lower left", bbox_to_anchor=[1, 0, 0.3, 0.3])
    if wrap:
        axs[-1].set_xticks(np.arange(0, 60 * 25, 3 * 60))

    axs[-1].set_xlabel("Time of the day")

    fig.tight_layout(pad=0.25, h_pad=0.1, w_pad=0.025)
    plt.subplots_adjust(hspace=0.1)

    return axs


def paired_violin_plot(data, y_label, x_label=None, x_field=None, y_field=None, hue=None, **kwargs):
    if x_label is None:
        x_label = "Activities"

    if x_field is None:
        x_field = "activity"

    if y_field is None:
        y_field = "duration"

    if hue is None:
        hue = "Source"

    kwargs_default = dict(scale="area", split=True, cut=0,
                          inner="quartile", palette="Set2")
    kwargs_default.update(kwargs)

    ax: plt.Axes | None = kwargs.get("ax", None)  # ax: Axes
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    orient = kwargs_default.get("orient", "v")
    xscale = kwargs_default.pop("xscale", None)
    yscale = kwargs_default.pop("yscale", None)
    if orient == "h":
        y_field, x_field = x_field, y_field
        y_label, x_label = tuple([x_label, y_label])

    ax = sb.violinplot(x=x_field, y=y_field, data=data, hue=hue, **kwargs_default)
    ax.set_ylabel(y_label)

    if xscale is not None:
        ax.set_xscale(xscale)
        ax.set_xlim(1, 24 * 60)

    if yscale is not None:
        ax.set_yscale(yscale)
        ax.set_ylim(1, 24 * 60)

    if orient == "v":
        ax.yaxis.set_major_formatter(min2timestr)
    else:
        ax.xaxis.set_major_formatter(min2timestr)

    ax.set_xlabel(x_label)

    if "ax" not in kwargs:
        fig.tight_layout()

    return ax


def plot_edge_variables(dataframes, nx_metric="degree", epi_metric=None, averages=True, avg_by="source"):
    c1, c2, c3, c4 = [v["color"] for v in plt.rcParams["axes.prop_cycle"][:4]]
    subplots = []
    if epi_metric is None:
        epi_metric = ["infection time"]

    hue_order_index = {c.split(" ")[0].lower(): c for c in dataframes["edges"].type.cat.categories}
    hue_order = [hue_order_index[c] for c in ['family', 'friend', 'acquaintance', 'random']]

    out = Output()
    with out:
        jg = plot_grid_joint(dataframes["edges"],
                             y_vars=["infection time", "infection time"],
                             x_vars=[["degree", "strength"],
                                     ["degree_target", "strength_target"]],
                             hue_order=hue_order, palette=[c1, c2, c3, c4])
        display(jg.figure)
        # plt.close(jg.figure)

    subplots.append(out)

    # for weighted, dfs in dataframes.items():
    #     for e_m in epi_metric:
    #         title = f"{e_m.title()}[d] - {'Weighted' if weighted else 'Unweighted'}"
    #         data = dfs["edges"]
    #
    #         if averages:
    #             data = (data
    #                     .set_index([avg_by, "type"], append=True)
    #                     .groupby(level=[0, avg_by, "type"])
    #                     .mean()
    #                     )

    return subplots


class ExtendedJointGrid:
    def __init__(self, x_vars, y_vars, fig: Figure = None, subplot_spec: SubplotSpec = None, groupby_index=None):
        self.x_vars = np.array(x_vars)
        self.y_vars = np.array(y_vars)

        self.figure = None
        self.subplot_spec = None
        if fig is not None:
            self.figure = fig

        if subplot_spec is not None:
            self.subplot_spec = subplot_spec
            self.figure = subplot_spec.get_gridspec().figure

        self.grid_shape = np.array([2, 2])

        self.joint_axes = []
        self.marginal_axes = []

        self.auto_arrange_grid()
        self.beautyfy()
        # self.figure.tight_layout()
        # self.figure.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.2, left=0.073, right=0.99)

    def auto_arrange_grid(self):
        if self.x_vars.shape == self.y_vars.shape and self.x_vars.ndim > 1:
            self.__independent_grid__()

        elif self.x_vars.ndim == self.y_vars.ndim == 1:
            variable_pairs = product(self.x_vars, self.y_vars)
            self.grid_shape[:] = len(self.x_vars) + 1, len(self.y_vars) + 1
            share_x = True
            share_y = True

        elif self.x_vars.ndim == 1 and len(self.x_vars) == self.y_vars.shape[1]:
            variable_pairs = [(xv, yv) for idx, xv in enumerate(self.x_vars) for yv in self.y_vars[:, idx]]
            self.grid_shape[:] = np.array(self.y_vars.shape) + 1
            share_x = True

        elif self.y_vars.ndim == 1 and len(self.y_vars) == self.x_vars.shape[0]:
            self.__shared_y_axis()

        else:
            raise AssertionError("Cannot decipher grid structure from x_vars and y_vars")

    def __independent_grid__(self):
        # Assume a specific x and y variable were provided for each grid slot.
        # For each joint distribution we have to plot corresponding marginal distributions
        self.grid_shape[:] = self.x_vars.shape
        self.global_gs = self.figure.add_gridspec(*self.grid_shape * [6, 1])
        for x_var, y_var, gs in zip(self.x_vars.ravel(), self.y_vars.ravel(), self.global_gs):
            local_gs = gs.subgridspec(6, 6)
            joint_ax = self.figure.add_subplot(local_gs[1:, :-1])
            marginal_x_ax = self.figure.add_subplot(local_gs[0, :-1], sharex=joint_ax)
            marginal_y_ax = self.figure.add_subplot(local_gs[1:, -1], sharey=joint_ax)

            self.joint_axes.append((x_var, y_var, joint_ax))
            self.marginal_axes.extend([("x", x_var, marginal_x_ax), ("y", y_var, marginal_y_ax)])

    def __shared_y_axis(self):
        # Each row uses 5 slots for the joint plot and one slot for th x marginal distribution.
        # Each column uses 5 slots, the last column is one slot wide to store the y marginal distribution.
        self.grid_shape[:] = self.y_vars.shape[0] * 6, self.x_vars.shape[1] * 5 + 1
        self.global_gs = self.figure.add_gridspec(*self.grid_shape)

        variable_pairs = [(xv, yv) for idx, yv in enumerate(self.y_vars) for xv in self.x_vars[idx, :]]
        x_marginal_coordinates = sorted([(i*6, slice(j*5, (j+1) * 5)) for i, j in
                                         product(range(self.y_vars.shape[0]), range(self.x_vars.shape[1]))],
                                        key=itemgetter(1))
        y_marginal_coordinates = [(slice(i*6 + 1, (i+1) * 6), -1) for i in range(self.y_vars.shape[0])]

        # Create the coordinate pairs ensuring that we traverse the columns first.
        joint_coordinates = sorted([(slice(i*6 + 1, (i+1) * 6), slice(j*5, (j+1) * 5)) for i, j in
                                    product(range(self.y_vars.shape[0]), range(self.x_vars.shape[1]))],
                                   key=itemgetter(1))

        joint_axes = np.full((self.y_vars.shape[0], self.x_vars.shape[1]), None)

        # Create joint plots sharing the y-axis, we assume traversing columns first.
        first_plot = None
        for (x_var, y_var), gs_coordinates in zip(variable_pairs, joint_coordinates):
            sub_plot_spec = self.global_gs[gs_coordinates]
            row, col = (gs_coordinates[0].start - 1) // 6, gs_coordinates[1].start // 5

            if sub_plot_spec.is_first_col():
                first_plot = self.figure.add_subplot(sub_plot_spec)
                joint_axes[row, col] = first_plot
                self.joint_axes.append((x_var, y_var, first_plot))
                continue

            joint_axes[row, col] = self.figure.add_subplot(sub_plot_spec, sharey=first_plot)
            joint_axes[row, col].yaxis.label.set_visible(False)
            joint_axes[row, col].yaxis.set_tick_params(labelleft=False)
            self.joint_axes.append((x_var, y_var, joint_axes[row, col]))
            if gs_coordinates[0].stop == self.x_vars.shape[1] * 5:
                first_plot = None

        # create marginal x plots
        for (x_var, y_var), gs_coordinates in zip(variable_pairs, x_marginal_coordinates):
            row, col = gs_coordinates[0] // 6, gs_coordinates[1].start // 5
            marginal_x_ax = self.figure.add_subplot(self.global_gs[gs_coordinates], sharex=joint_axes[row, col])
            self.marginal_axes.append(("x", x_var, marginal_x_ax))

        # create marginal y plots
        for y_var, gs_coordinates in zip(self.y_vars, y_marginal_coordinates):
            row, col = (gs_coordinates[0].start - 1) // 6, -1
            marginal_y_ax = self.figure.add_subplot(self.global_gs[gs_coordinates], sharey=joint_axes[row, 0])
            self.marginal_axes.append(("y", y_var, marginal_y_ax))

    def __shared_x_axis(self):
        pass

    def __shared_both_axis(self):
        pass

    def beautyfy(self):
        for x_var, y_var, joint_ax in self.joint_axes:
            despine(ax=joint_ax)

        for ax_type, x_var, marginal_ax in self.marginal_axes:
            if ax_type == "x":
                plt.setp(marginal_ax.get_xticklabels(), visible=False)
                plt.setp(marginal_ax.get_xticklabels(minor=True), visible=False)

                # Turn off ticks for density plots
                plt.setp(marginal_ax.yaxis.get_majorticklines(), visible=False)
                plt.setp(marginal_ax.yaxis.get_minorticklines(), visible=False)
                plt.setp(marginal_ax.get_yticklabels(), visible=False)
                plt.setp(marginal_ax.get_yticklabels(minor=True), visible=False)
                marginal_ax.yaxis.grid(False)
                despine(ax=marginal_ax, left=True)

            else:

                plt.setp(marginal_ax.get_yticklabels(), visible=False)
                plt.setp(marginal_ax.get_yticklabels(minor=True), visible=False)
                plt.setp(marginal_ax.xaxis.get_majorticklines(), visible=False)
                plt.setp(marginal_ax.xaxis.get_minorticklines(), visible=False)
                plt.setp(marginal_ax.get_xticklabels(), visible=False)
                plt.setp(marginal_ax.get_xticklabels(minor=True), visible=False)
                marginal_ax.xaxis.grid(False)
                despine(ax=marginal_ax, bottom=True)

            for axis in [marginal_ax.xaxis, marginal_ax.yaxis]:
                axis.label.set_visible(False)


def plot_grid_joint(data, x_vars, y_vars, groupby_index=None, hue=None, hue_order=None, title=None, palette=None,
                    legend_kwargs=None, add_counts=True, fig=None, subplot_spec=None, fig_size=None):
    if fig_size is None:
        fig_size = (10, 3)

    if fig is None and subplot_spec is None:
        fig = plt.figure(figsize=fig_size)

    if legend_kwargs is None:
        legend_kwargs = {"loc": "lower left",
                         "bbox_to_anchor": (0.3, 0.5)}

    extended_joint_grid = ExtendedJointGrid(x_vars, y_vars, fig, subplot_spec=subplot_spec, groupby_index=groupby_index)

    legend = True
    handles, labels = [], []
    for x_v, y_v, ax in extended_joint_grid.joint_axes:
        common_parameters = dict(x=x_v, y=y_v, data=data,
                                 )

        sb.scatterplot(ax=ax, **common_parameters, alpha=0.5, hue=hue, hue_order=hue_order,
                       palette=palette, legend=legend, s=5)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            l = ax.get_legend()
            if l is not None:
                l.set(visible=False)

        sb.regplot(ax=ax, scatter=False, **common_parameters)
        ax.set_xscale("log")

    for ax_type, marginal_var, marginal_ax, in extended_joint_grid.marginal_axes:
        common_joint_parameters = dict(data=data, cut=0,
                                       multiple="stack", alpha=0.5, lw=0.5, common_norm=True, legend=False,
                                       **{ax_type: marginal_var})
        if hue is not None:
            common_joint_parameters.update(hue=hue, hue_order=hue_order, palette=palette, ec="w")
        sb.kdeplot(ax=marginal_ax, **common_joint_parameters)

    if add_counts and hue is not None:
        hue_count = data.groupby(hue).size().to_dict()
        # labels = [f"{l.title()} ({hue_count[l] / sum(hue_count.values()) * 100: >2.0f}%)" for l in labels]
        labels = [f"{l.title()} (n={hue_count[l]:n})".replace(".", ",") for l in labels]

    if hue is not None:
        title = f"{hue.title()}" if not add_counts else f"{hue.title()} (Samples: {len(data)})"
        title = None
        fig.legend(handles, labels, **legend_kwargs, title=title)

    return fig


def plot_vertex_variables(dataframes, epi_metrics=None, nx_metric="degree"):
    c1, c2, c3, c4 = [v["color"] for v in plt.rcParams["axes.prop_cycle"][:4]]
    subplots = []

    if epi_metrics is None:
        epi_metrics = ["infected"]

    for weighted, dfs in dataframes.items():
        for e_m in epi_metrics:
            title = f"{e_m.title()} - {'Weighted' if weighted else 'Unweighted'}"
            data = pd.concat(dfs["vertices"], keys=range(len(dfs)))

            out = Output()
            with out:
                jg = sb.jointplot(x=e_m, y=nx_metric, data=data, hue="type",
                                  hue_order=['family', 'friend', 'acquaintance', 'random'],
                                  palette=[c1, c2, c4, c3],
                                  marginal_ticks=True,
                                  marginal_kws={"cut": 0, "common_norm": True, "multiple": "layer"})

                jg.ax_joint.set_ylabel(nx_metric.title())
                jg.ax_joint.set_xlabel(title)
                jg.figure.tight_layout()
                display(jg.figure)

            subplots.append(out)

            out = Output()
            with out:
                jg = sb.jointplot(x=e_m, y=nx_metric, data=data.groupby(level=[0, 1], axis=0).sum(),
                                  marginal_ticks=True,
                                  marginal_kws={"kde": True})

                # jg.plot_marginals(sb.kdeplot, cut=0, common_norm=False )

                jg.ax_joint.set_ylabel(nx_metric.title())
                jg.ax_joint.set_xlabel(f"Total {title}")
                jg.figure.tight_layout()
                display(jg.figure)
                # plt.close(jg.figure)

            subplots.append(out)

    return subplots
