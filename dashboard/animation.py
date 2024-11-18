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
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import time
from matplotlib import cbook
from matplotlib.animation import FuncAnimation, writers, PillowWriter

from dashboard.plot_utils import _val_or_rc


class FuncAnimationFullBlit(FuncAnimation):
    def _blit_draw(self, artists):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure.
        updated_ax = {a.axes for a in artists}
        # Enumerate artists to cache Axes backgrounds. We do not draw
        # artists yet to not cache foreground from plots with shared Axes
        for ax in updated_ax:
            # If we haven't cached the background for the current view of this
            # Axes object, do so now. This might not always be reliable, but
            # it's an attempt to automate the process.
            cur_view = ax._get_view()
            view, bg = self._blit_cache.get(ax, (object(), None))
            if cur_view != view:
                bbox = ax.bbox
                if ax.xaxis.get_animated():
                    bbox = bbox.union([bbox, ax.xaxis.get_tightbbox()])

                if ax.yaxis.get_animated():
                    bbox = bbox.union([bbox, ax.yaxis.get_tightbbox()])

                self._blit_cache[ax] = (
                    cur_view, ax.figure.canvas.copy_from_bbox(bbox))

        # Make a separate pass to draw foreground.
        for a in artists:
            a.axes.draw_artist(a)

        # After rendering all the needed artists, blit each Axes individually.
        for ax in updated_ax:
            bbox = ax.bbox
            if ax.xaxis.get_animated():
                bbox = bbox.union([bbox, ax.xaxis.get_tightbbox()])

            if ax.yaxis.get_animated():
                bbox = bbox.union([bbox, ax.yaxis.get_tightbbox()])

            ax.figure.canvas.blit(bbox)

    def save(self, filename, writer=None, fps=None, dpi=None, codec=None,
             bitrate=None, extra_args=None, metadata=None, extra_anim=None,
             savefig_kwargs=None, *, progress_callback=None):
        """
        Save the animation as a movie file by drawing every frame.

        Parameters
        ----------
        filename : str
            The output filename, e.g., :file:`mymovie.mp4`.

        writer : `MovieWriter` or str, default: :rc:`animation.writer`
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        fps : int, optional
            Movie frame rate (per second).  If not set, the frame rate from the
            animation's frame interval.

        dpi : float, default: :rc:`savefig.dpi`
            Controls the dots per inch for the movie frames.  Together with
            the figure's size in inches, this controls the size of the movie.

        codec : str, default: :rc:`animation.codec`.
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

        bitrate : int, default: :rc:`animation.bitrate`
            The bitrate of the movie, in kilobits per second.  Higher values
            means higher quality movies, but increase the file size.  A value
            of -1 lets the underlying movie encoder select the bitrate.

        extra_args : list of str or None, optional
            Extra command-line arguments passed to the underlying movie encoder. These
            arguments are passed last to the encoder, just before the output filename.
            The default, None, means to use :rc:`animation.[name-of-encoder]_args` for
            the builtin writers.

        metadata : dict[str, str], default: {}
            Dictionary of keys and values for metadata to include in
            the output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.

        extra_anim : list, default: []
            Additional `Animation` objects that should be included
            in the saved movie file. These need to be from the same
            `.Figure` instance. Also, animation frames will
            just be simply combined, so there should be a 1:1 correspondence
            between the frames from the different animations.

        savefig_kwargs : dict, default: {}
            Keyword arguments passed to each `~.Figure.savefig` call used to
            save the individual frames.

        progress_callback : function, optional
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and *total_frames* is the
            total number of frames to be saved. *total_frames* is set to None, if the
            total number of frames cannot be determined. Return values may exist but are
            ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')

        Notes
        -----
        *fps*, *codec*, *bitrate*, *extra_args* and *metadata* are used to
        construct a `.MovieWriter` instance and can only be passed if
        *writer* is a string.  If they are passed as non-*None* and *writer*
        is a `.MovieWriter`, a `RuntimeError` will be raised.
        """

        all_anim = [self]
        if extra_anim is not None:
            all_anim.extend(anim for anim in extra_anim
                            if anim._fig is self._fig)

        # Disable "Animation was deleted without rendering" warning.
        for anim in all_anim:
            anim._draw_was_started = True

        if writer is None:
            writer = mpl.rcParams['animation.writer']
        elif (not isinstance(writer, str) and
              any(arg is not None
                  for arg in (fps, codec, bitrate, extra_args, metadata))):
            raise RuntimeError('Passing in values for arguments '
                               'fps, codec, bitrate, extra_args, or metadata '
                               'is not supported when writer is an existing '
                               'MovieWriter instance. These should instead be '
                               'passed as arguments when creating the '
                               'MovieWriter instance.')

        if savefig_kwargs is None:
            savefig_kwargs = {}
        else:
            # we are going to mutate this below
            savefig_kwargs = dict(savefig_kwargs)

        if fps is None and hasattr(self, '_interval'):
            # Convert interval in ms to frames per second
            fps = 1000. / self._interval

        # Reuse the savefig DPI for ours if none is given.
        dpi = _val_or_rc(dpi, 'savefig.dpi')
        if dpi == 'figure':
            dpi = self._fig.dpi

        writer_kwargs = {}
        if codec is not None:
            writer_kwargs['codec'] = codec
        if bitrate is not None:
            writer_kwargs['bitrate'] = bitrate
        if extra_args is not None:
            writer_kwargs['extra_args'] = extra_args
        if metadata is not None:
            writer_kwargs['metadata'] = metadata

        # If we have the name of a writer, instantiate an instance of the
        # registered class.
        if isinstance(writer, str):
            try:
                writer_cls = writers[writer]
            except RuntimeError:  # Raised if not available.
                writer_cls = PillowWriter  # Always available.
                mpl._log.warning("MovieWriter %s unavailable; using Pillow "
                             "instead.", writer)
            writer = writer_cls(fps, **writer_kwargs)
        mpl._log.info('Animation.save using %s', type(writer))

        if 'bbox_inches' in savefig_kwargs:
            mpl._log.warning("Warning: discarding the 'bbox_inches' argument in "
                         "'savefig_kwargs' as it may cause frame size "
                         "to vary, which is inappropriate for animation.")
            savefig_kwargs.pop('bbox_inches')

        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't work
        # since GUI widgets are gone. Either need to remove extra code to
        # allow for this non-existent use case or find a way to make it work.

        facecolor = savefig_kwargs.get('facecolor',
                                       mpl.rcParams['savefig.facecolor'])
        if facecolor == 'auto':
            facecolor = self._fig.get_facecolor()

        def _pre_composite_to_white(color):
            r, g, b, a = mcolors.to_rgba(color)
            return a * np.array([r, g, b]) + 1 - a

        savefig_kwargs['facecolor'] = _pre_composite_to_white(facecolor)
        savefig_kwargs['transparent'] = False  # just to be safe!
        # canvas._is_saving = True makes the draw_event animation-starting
        # callback a no-op; canvas.manager = None prevents resizing the GUI
        # widget (both are likewise done in savefig()).
        with writer.saving(self._fig, filename, dpi), \
                cbook._setattr_cm(self._fig.canvas, _is_saving=True, manager=None):
            for anim in all_anim:
                anim._init_draw()  # Clear the initial frame
            frame_number = 0
            # TODO: Currently only FuncAnimation has a save_count
            #       attribute. Can we generalize this to all Animations?
            save_count_list = [getattr(a, '_save_count', None)
                               for a in all_anim]

            if None in save_count_list:
                total_frames = None
            else:
                total_frames = sum(save_count_list)

            for data in zip(*[a.new_saved_frame_seq() for a in all_anim]):
                t0 = time.time_ns()
                frame_number += 1
                for anim, d in zip(all_anim, data):
                    # TODO: See if turning off blit is really necessary
                    anim._draw_next_frame(d, blit=True)
                    if progress_callback is not None:
                        progress_callback(frame_number, total_frames)

                t_update = (time.time_ns() - t0)/1e9

                t0 = time.time_ns()
                self._fig.canvas.flush_events()
                renderer = self._fig.canvas.get_renderer()
                with cbook.open_file_cm(writer._proc.stdin, "wb") as fh:
                    fh.write(renderer.buffer_rgba())

                t_grab = (time.time_ns() - t0)/1.e9

                # renderer.clear()

                print(f"\rWirting frame: {frame_number}, Darwing: {t_update:0.2f}, Saving: {t_grab:0.2f}", end="")
