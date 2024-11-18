import numbers

import numpy as np
from itertools import cycle
from matplotlib import rcParams, _tight_layout
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform


class ExtendedPathPatch(PathPatch):
    """Extends PathPatch to support multiple axes.
     Current use case is to draw shaded areas in parallel coordinates."""

    def __init__(self, path, transforms, **kwargs):
        """path is a collection of points in data coordinates, for each point there is the associated axes."""

        self.data_path = path.copy()
        self.transformers = transforms
        super().__init__(path, **kwargs)

        path = self._get_path_in_displaycoord()
        self.set_path(path)

    def _get_path_in_displaycoord(self):
        vertices = np.array([trans.transform(p) for p, trans in zip(self.data_path, self.transformers)])
        codes = np.full(len(vertices), Path.LINETO)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        area_boundary = Path(vertices=vertices, codes=codes)

        return area_boundary

    def get_transform(self):
        return IdentityTransform()

    def draw(self, renderer):
        if renderer is not None:
            self._renderer = renderer

        if not self.get_visible():
            return

        # FIXME: dpi_cor is for the dpi-dependency of the linewidth.  There
        # could be room for improvement.  Maybe _get_path_in_displaycoord could
        # take a renderer argument, but get_path should be adapted too.
        self._dpi_cor = renderer.points_to_pixels(1.)
        path = self._get_path_in_displaycoord()
        affine = IdentityTransform()
        self._draw_paths_with_artist_properties(
            renderer,
            [(path, affine,
              # Work around a bug in the PDF and SVG renderers, which
              # do not draw the hatches if the facecolor is fully
              # transparent, but do if it is None.
              self._facecolor if self._facecolor[3] else None)])

    def get_path(self):
        return self._get_path_in_displaycoord()

class ParallelCoordinates:
    def __init__(self, fig: Figure, axes_labels, grid_spec=None, rot=0., sharedx=False):

        self.axes_labels = axes_labels
        if grid_spec is None:
            grid_spec = fig.add_gridspec(1, len(axes_labels), wspace=0, hspace=0, left=0, right=1., top=1., bottom=0,
                                         wratios=[0.1] * len(axes_labels))
            self.grid_spec = grid_spec
        else:
            grid_spec = grid_spec.subgridspec(1, len(axes_labels), wspace=0, hspace=0)
            self.grid_spec = grid_spec

        self.fig = fig
        self.n = len(axes_labels)
        self.axes = grid_spec.subplots()
        self.ax = self.axes[0]

        for ax, title in zip(self.axes, axes_labels):
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines.left.set_position("center")
            ax.set_xticks([0])
            if not sharedx:
                ax.set_xticklabels([title], rotation=rot)
            else:
                ax.set_xticklabels([])

    def fill_between(self, path1, path2, *args, **kw):
        kw = kw.copy()
        color_cycle = self.__get_color_cycle(kw)
        lw_cycle = self.__get_lineweight_cycle(kw)
        ls_cycle = self.__get_linestyle_cycle(kw)
        path1 = np.column_stack([np.zeros_like(path1), np.r_[path1]])
        ax_transforms1 = [ax.transData for ax in self.axes]

        # Extend the path with the second path inverted. Invert also transformers so that each point corresponds to
        # the correct transformer.
        path2 = np.column_stack([np.zeros_like(path2), path2.ravel()])[::-1]
        ax_transforms2 = [ax.transData for ax in self.axes[::-1]]

        # # Close the polygon
        path = np.vstack([path1, path2, [path1[0, :]]])
        ax_transforms = ax_transforms1 + ax_transforms2 + ax_transforms1[0:1]

        color = kw.get("color", next(color_cycle))

        self.fig.add_artist(ExtendedPathPatch(path, ax_transforms, facecolor=color, alpha=0.3))

    def plot(self, data, *args, **kw):
        kw = kw.copy()
        color_cycle = self.__get_color_cycle(kw)
        lw_cycle = self.__get_lineweight_cycle(kw)
        ls_cycle = self.__get_linestyle_cycle(kw)
        for ix, row in data.iterrows():
            values = np.r_[row]
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

        title = kw.pop("title", None)
        if title is not None:
            self.set_title(title)

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

    def set_title(self, title):
        self.ax.set_title(title, pad=0)
        self.ax.title.set(fontweight="bold", ha="left")

    def legend(self, *args, **kwargs):
        bbox_ = np.array([-len(self.axes_labels) + 1.5, -0.3, len(self.axes_labels) - 1, 0.02])
        mode = kwargs.pop("mode", "expand")
        loc = kwargs.pop("loc", "best")
        bbox = kwargs.pop("bbox_to_anchor", [0, 0, 0, 0])
        bbox_[:len(bbox)] += bbox
        kwargs.update(dict(loc=loc, mode=mode, borderaxespad=0.0,
                           bbox_to_anchor=bbox_),
                      )
        l = self.axes[-1].legend(*args, **kwargs)
        # l.set(clip_on=False, zorder=1000)
        l.set_in_layout(True)

        return l

    def tight_layout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        subplotspec_list = _tight_layout.get_subplotspec_list(
            self.axes, grid_spec=self.grid_spec)

        renderer = self.fig._get_renderer()

        kwargs = _tight_layout.get_tight_layout_figure(
            self.fig, self.axes, subplotspec_list, renderer,
            pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        self.update_gridspec(**kwargs)

    _AllowedKeys = ["left", "bottom", "right", "top", "wspace", "hspace"]

    def update_gridspec(self, **kwargs):
        """
        Update the subplot parameters of the grid.
        Parameters that are not explicitly given are not changed. Setting a
        parameter to *None* resets it to :rc:`figure.subplot.*`.
        Parameters
        ----------
        left, right, top, bottom : float or None, optional
            Extent of the subplots as a fraction of figure width or height.
        wspace, hspace : float, optional
            Spacing between the subplots as a fraction of the average subplot
            width / height.
        """
        for k, v in kwargs.items():
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError(f"{k} is an unknown keyword")

        for ax in self.axes:
            ax._set_position(
                ax.get_subplotspec().get_position(ax.figure))
