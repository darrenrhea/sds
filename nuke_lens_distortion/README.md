# nuke_lens_distortion


## Run Tests

```
pytest -s
```

In the Nuke lens distortion model, the map from
distorted normalized screen coordinates in [-1, 1] x [-9/16, 9/16] (or whatever your aspect ratio is)
to undistorted screen coordinates is given by a polynomial
(is closed form, thus cheap to calculatee), whereas the other direction,
from undistorted to distorted, is not closed form and thus has to be solved
by Newton's method etc.  ``nuke_lens_distortion`` handles that.

