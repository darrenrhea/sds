
# Drawable2DImage

Drawable2DImage is a Python package to draw line segments and
parametric curves (circle arcs is a common use),
annotations with text labels.



## Installation

```
pip install Drawable2DImage
```

Or if you have clone access:

```
git clone git@github.com:darrenrhea/Drawable2DImage.git
cd Drawable2DImage
pip install -e .
```

## Demo

Once installed, type

```
demo_Drawable2DImage
open out.png
```

## Performance

In general, writing a `.bmp` file is wildly faster than writing a `.png` file.

Also, if you are going to write pngs, turn compression off to save many seconds per save.

When writing a 3840x2160 png, we go from 6.5 seconds to 1.5 seconds by turning .png compression level during save to 1:
Over a thousand images, which is only 16.6 seconds of 60 fps video,
that is 4000 seconds, i.e 1 hour 6 minutes 40 seconds of savings.

```python
def save(self, output_image_file_path):
        output_image_file_path = Path(output_image_file_path).expanduser()
        self.image_pil.save(
            fp=str(output_image_file_path),
            format="PNG",
            compress_level=1  # this is speed critical.
        )  
```

Not too surprisingly, 1920x1080 is faster:

* 0.430 seconds for 1920 x 1080  `.png` `compress_level=1`, and temp.png is 3.1 MBytes
* 1.637 seconds for 1920 x 1080 `.png` compress_level unspecified, temp.png is 2.6 Mbytes

* 1.53 seconds for 3840 x 2160 `.png` `compress_level=1`, temp.png is 9.5 Megabytes
* 6.55 seconds for 3840 x 2160 `.png` compress_level unspecified, temp.png is 7.7 Megabytes

bmp is faster by far, but bigger:

* 0.006 seconds for 1920 x 1080 `.bmp`, temp.png is 8.3 Megabytes
* 0.035 seconds for 3840 x 2160 `.bmp`, temp.png is 33.2 Megabytes
