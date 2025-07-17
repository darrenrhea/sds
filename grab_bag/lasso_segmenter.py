from collections.abc import Callable

from mpl_interactions import panhandler, zoom_factory, image_segmenter

import numpy as np
from matplotlib import __version__ as mpl_version
from matplotlib import get_backend
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS, to_rgba_array
from matplotlib.path import Path
from matplotlib.pyplot import close, ioff, subplots
from matplotlib.widgets import LassoSelector
import matplotlib.pyplot as plt
from numpy import asanyarray, asarray, max, min
import cv2

from mpl_interactions.controller import gogogo_controls, prep_scalars
from mpl_interactions.helpers import (
    callable_else_value_no_cast,
    create_slider_format_dict,
    gogogo_figure,
    notebook_backend,
)
from mpl_interactions.mpl_kwargs import imshow_kwargs_list, kwarg_popper
from mpl_interactions.utils import figure, nearest_idx
from mpl_interactions.xarray_helpers import get_hs_axes, get_hs_extent, get_hs_fmts

CLOSE_KERNEL_SIZE = 5

myindices = None
myws = None

class my_image_segmenter(image_segmenter):
    """Manually segment an image with the lasso selector."""

    def __init__(
        self,
        img,
        nclasses=1,
        mask=None,
        mask_colors=None,
        mask_alpha=0.75,
        lineprops=None,
        props=None,
        lasso_mousebutton="left",
        pan_mousebutton="middle",
        ax=None,
        figsize=(10, 10),
        filename = None,
        **kwargs,
    ):
        """Create an image segmenter.

        Any ``kwargs`` will be passed through to the ``imshow``
        call that displays *img*.

        Parameters
        ----------
        img : array_like
            A valid argument to imshow
        nclasses : int, default 1
            How many classes.
        mask : arraylike, optional
            If you want to pre-seed the mask
        mask_colors : None, color, or array of colors, optional
            the colors to use for each class. Unselected regions will always be totally transparent
        mask_alpha : float, default .75
            The alpha values to use for selected regions. This will always override the alpha values
            in mask_colors if any were passed
        lineprops : dict, default: None
            DEPRECATED - use props instead.
            lineprops passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        props : dict, default: None
            props passed to LassoSelector. If None the default values are:
            {"color": "black", "linewidth": 1, "alpha": 0.8}
        lasso_mousebutton : str, or int, default: "left"
            The mouse button to use for drawing the selecting lasso.
        pan_mousebutton : str, or int, default: "middle"
            The button to use for `~mpl_interactions.generic.panhandler`. One of 'left', 'middle' or
            'right', or 1, 2, 3 respectively.
        ax : `matplotlib.axes.Axes`, optional
            The axis on which to plot. If *None* a new figure will be created.
        figsize : (float, float), optional
            passed to plt.figure. Ignored if *ax* is given.
        **kwargs
            All other kwargs will passed to the imshow command for the image
        """
        # ensure mask colors is iterable and the same length as the number of classes
        # choose colors from default color cycle?

        self.mask_alpha = mask_alpha

        if mask_colors is None:
            # this will break if there are more than 10 classes
            if nclasses <= 10:
                self.mask_colors = to_rgba_array(list(TABLEAU_COLORS)[:nclasses])
            else:
                # up to 949 classes. Hopefully that is always enough....
                self.mask_colors = to_rgba_array(list(XKCD_COLORS)[:nclasses])
        else:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            # should probably check the shape here
        self.mask_colors[:, -1] = self.mask_alpha

        self.nclasses = nclasses

        if ax is not None:
            self.ax = ax
            self.fig = self.ax.figure
        else:
            with ioff():
                self.fig = figure(figsize=figsize)
                self.ax = self.fig.gca()

        self.mode = 'add'
        self.set_image(img, mask = mask, fn = filename)

        default_props = self.MODES['add']

        if (props is None) and (lineprops is None):
            props = default_props
        elif (lineprops is not None) and (mpl_version >= "3.7"):
            print("*lineprops* is deprecated in matplotlib 3.7+,  please use *props*")
            props = default_props

        self.useblit = False if "ipympl" in get_backend().lower() else True

        button_dict = {"left": 1, "middle": 2, "right": 3}
        if isinstance(pan_mousebutton, str):
            pan_mousebutton = button_dict[pan_mousebutton.lower()]

        if isinstance(lasso_mousebutton, str):
            lasso_mousebutton = button_dict[lasso_mousebutton.lower()]

        self.lasso_mousebutton = lasso_mousebutton

        if mpl_version < "3.7":
            self.lasso = LassoSelector(
                self.ax, self._onselect, lineprops=props, useblit=self.useblit, button=self.lasso_mousebutton
            )

        else:
            self.lasso = LassoSelector(
                self.ax, self._onselect, props=props, useblit=self.useblit, button=self.lasso_mousebutton
            )
        self.lasso.set_visible(True)

        pix_x = np.arange(self._img.shape[0])
        pix_y = np.arange(self._img.shape[1])
        xv, yv = np.meshgrid(pix_y, pix_x)
        self.pix = np.vstack((xv.flatten(), yv.flatten())).T

        self.ph = panhandler(self.fig, button=pan_mousebutton)

        self.disconnect_zoom = zoom_factory(self.ax)

        self.current_class = 1

        self.erasing = False

        self.undo_stack = []
        self.redo_stack = []

        self._is_dirty = False

        # copy + paste
        self.copy_mask = None
        self.copy_ind = None

    def watershed(self, indices):
        mask = indices.astype(np.uint8)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            print('no contours!')
            return

        x, y, w, h = cv2.boundingRect(cnts[0][0])

        mask[mask > 0] = cv2.GC_PR_FGD
        mask[mask == 0] = cv2.GC_BGD

        # M = cv2.moments(cnts[0][0])
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        # seed = (cX, cY)
        # print(seed)

        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        (outmask, bgModel, fgModel) = cv2.grabCut(self._img, mask, None, bgModel,
            fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)

        masked = (outmask == cv2.GC_PR_FGD).astype(np.uint8)
        newmask = self.mask.copy()
        newmask[y:y+h, x:x+w] |= masked[y:y+h, x:x+w]
        self.set_mask(newmask)

    def set_dirty(self, state):
        self._is_dirty = state

    def is_dirty(self):
        return len(self.undo_stack) > 0 or self._is_dirty

    def reset_zoom(self):
        self.ax.set_xlim(*self.init_xlim)
        self.ax.set_ylim(*self.init_ylim)
        self.fig.canvas.draw()

    MODES = {
        'add': {"color": "green", "linewidth": 1, "alpha": 0.8},
        'del': {"color": "red", "linewidth": 1, "alpha": 0.8},
        'copy': {"color": "yellow", "linewidth": 1, "alpha": 0.8},
        'close': {"color": "cyan", "linewidth": 1, "alpha": 0.8},
        'watershed': {"color": "magenta", "linewidth": 1, "alpha": 0.8},
    }

    def set_mode(self, mode):
        if not mode in self.MODES:
            print(f'ERROR: unknown mode {mode}')
            return

        self.mode = mode

        if self.mode_text is None:
            self.mode_text = self.ax.text(0, -40, mode)
        else:
            self.mode_text.set_text(mode)

        try:
            self.lasso.set_props(**self.MODES[mode])
        except Exception as e:
            pass

        self.fig.canvas.draw()

    def save_mask(self, fn):
        if self.is_dirty() > 0:
            print(f'saving {fn}')
            frame_bgr = cv2.cvtColor(self._img, cv2.COLOR_RGB2BGR)
            frame_bgra = np.zeros((*frame_bgr.shape[:2], 4), dtype=np.uint8)
            frame_bgra[...,:3] = frame_bgr
            mask = self.mask.copy()
            mask[mask == 1] = 255
            frame_bgra[..., 3] = mask
            cv2.imwrite(fn, frame_bgra)
            return True
        return False

    def set_mask(self, mask):
        if not self.mask is None:
            if np.all(self.mask == mask):
                print('no mask change')
            else:
                print('mask changed')
                self.undo_stack.append([self.mask.copy(), self._overlay.copy()])

        print('loading mask!')
        self.mask = mask.copy()

        for i in range(self.nclasses + 1):
            idx = self.mask == i
            if i == 0:
                self._overlay[idx] = [0, 0, 0, 0]
            else:
                self._overlay[idx] = self.mask_colors[i - 1]

        self._mask.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    def get_lim(self):
        return self.ax.get_xlim(), self.ax.get_ylim()

    def set_lim(self, xlim, ylim):
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

    def set_image(self, img, mask = None, fn = None):
        self._img = np.asarray(img)
        self._overlay = np.zeros((*self._img.shape[:2], 4))

        self.ax.clear()

        self.displayed = self.ax.imshow(self._img)
        self._mask = self.ax.imshow(self._overlay)
        self.ax.set_axis_off()

        if mask is None:
            mask = np.zeros(self._img.shape[:2], dtype = int)

        self.mask = None
        self.set_mask(mask)

        self.undo_stack = []
        self.redo_stack = []

        self.init_xlim = self.ax.get_xlim()
        self.init_ylim = self.ax.get_ylim()

        if fn:
            self.filename = self.ax.text(0, -20, fn)

        self.mode_text = self.ax.text(0, -40, self.mode)

        self.fig.canvas.draw()

    def set_mask_color(self, mask_colors = None, alpha = None):
        if not alpha is None and alpha >= 0.0 and alpha <= 1.0:
            self.mask_alpha = alpha
            self.mask_colors[:, -1] = self.mask_alpha

        if not mask_colors is None:
            self.mask_colors = to_rgba_array(np.atleast_1d(mask_colors))
            self.mask_colors[:, -1] = self.mask_alpha

        cc = self.mask == self.current_class
        self._overlay[cc] = self.mask_colors[self.current_class - 1]
        self._overlay[~cc] = [0, 0, 0, 0]
        self._mask.set_data(self._overlay)
        self.fig.canvas.draw()

    def _push_undo(self):
        self.redo_stack = []
        self.undo_stack.append([self.mask.copy(), self._overlay.copy()])

    def _onselect(self, verts):
        self.verts = verts
        p = Path(verts)
        self.indices = p.contains_points(self.pix, radius=0).reshape(self.mask.shape)

        if self.indices.sum() == 0:
            print('nothing selected')
            self._overlay[self.indices] = [0, 0, 0, 0]
            self._mask.set_data(self._overlay)
            self.fig.canvas.draw_idle()
            return

        modified = False
        if self.mode == 'copy':
            print('selection copied!')
            self.mode_text.set_text('selection copied!')
            self.copy_mask = self.mask.copy()
            self.copy_ind = self.indices.copy()
        elif self.mode == 'add' or self.mode == 'del':
            self._push_undo()
            modified = True

            if self.mode == 'del':
                self.mask[self.indices] = 0
                self._overlay[self.indices] = [0, 0, 0, 0]
            else:
                self.mask[self.indices] = self.current_class
                self._overlay[self.indices] = self.mask_colors[self.current_class - 1]
        elif self.mode == 'close':
            self._push_undo()
            modified = True
            self.close_mask(indices=self.indices)
        elif self.mode == 'watershed':
            self._push_undo()
            modified = True
            self.watershed(indices=self.indices)


        if modified:
            self._mask.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    def _message(self, text):
        print(text)
        if self.mode_text is None:
            self.mode_text = self.ax.text(0, -40, text)
        else:
            self.mode_text.set_text(text)

    def paste(self):
        if self.copy_mask is None:
            print('nothing copied, cannot paste!')
            return
        self._push_undo()
        mask = self.mask.copy()
        mask[self.copy_ind] = self.copy_mask[self.copy_ind]
        self.set_mask(mask)

    def undo(self):
        if len(self.undo_stack) < 1:
            print('nothing to undo!')
            return

        self.redo_stack.append([self.mask.copy(), self._overlay.copy()])

        action = self.undo_stack.pop()
        mask, _overlay = action

        self.mask[:, :] = mask
        self._overlay[:, :] = _overlay

        self._mask.set_data(self._overlay)
        self.fig.canvas.draw_idle()

    def redo(self):
        if len(self.redo_stack) < 1:
            print('nothing to redo!')
            return

        self.undo_stack.append([self.mask.copy(), self._overlay.copy()])

        action = self.redo_stack.pop()
        mask, _overlay = action

        self.mask[:, :] = mask
        self._overlay[:, :] = _overlay

        self._mask.set_data(self._overlay)
        self.fig.canvas.draw_idle()


    def close_mask(self, indices = None):

        if indices is None:
            xmin = ymin = 0
            xmax = self.mask.shape[1] - 1
            ymax = self.mask.shape[0] - 1
        else:
            # get bounding box
            rows = np.any(self.indices, axis=1)
            cols = np.any(self.indices, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))

        mask = self.mask.copy()

        mask_view = self.mask[ymin:ymax+1, xmin:xmax+1]
        mask_view_alt = cv2.morphologyEx(mask_view, cv2.MORPH_CLOSE, kernel)

        mask[ymin:ymax+1, xmin:xmax+1] = mask_view_alt

        self.set_mask(mask)

