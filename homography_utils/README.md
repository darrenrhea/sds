# Homography Utils

Utilities for homographies

Somehow get some keyframes, namely
swinne1 4613, 4700, 4800, 4900,
with named, known locations in the ground-floor {z_world = 0} annoted in them.
We use the good fit chaz_locked to do this, although it could be done by hand.
Get visual confirmation that is it not wrong:

```
ssh lam
cd ~/r/homography_utils
mkdir -p temp
python landmarks_from_cameras_for_homographies.py
for x in 4613 4700 4800 4900 ; do pri $(printf "temp/swinney1_%d.png" $x) ; done
```

Use the popsift engine to write homographies from each keyframe to each video frame
in the range 4600 to 11000:

```
cd ~/felix/popsift/build
time Linux-x86_64/match_keyframes /home/drhea/awecom/data/clips/swinney1/frames/swinney1_004613.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004700.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004800.jpg,/home/drhea/awecom/data/clips/swinney1/frames/swinney1_004900.jpg /home/drhea/awecom/data/clips/swinney1/frames/swinney1_%06d.jpg 4600 11000 out
```

Make them JSON.  Probably this should be done by the popsift code:

```
mkdir -p /home/drhea/awecom/data/clips/swinney1/homography_attempts/felix
python convert_felix_format_to_json.py
```


```
cd ~/r/homography_video_maker/build
cat ../confs/homography_felix_swinney1.json
```

Calculate the smoothed together homographies:

```
python smoothly_connect_homography_charts.py
```

We are going to make a viewable video via `homography_video_maker`:

Edit its config file:

```
cd ~/r/homography_video_maker/build
edit ../confs/homography_felix_swinney1.json
```

to look like this, 6 red coke logos, frames from 10000 to 11000:

```json
{
    "clip_id": "swinney1",
    "tracking_attempt_id": "felix",
    "masking_attempt_id": "",
    "ad_image": "~/awecom/data/ads/red_cokelogo.png",
    "first_frame": 10000,
    "last_frame":  11000,
    "insertion_attempt_id": "homography",
    "draw_wireframe": false,
    "save_frames": true,
    "ads": [
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 12.0,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -12.0,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": -11.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": -11.2,
            "width": 8,
            "height": 5
        }
    ]
}
```

Run it:

```
cd ~/r/homography_video_maker/build
export DISPLAY=:1
bin/ad_insertion ../confs/homography_felix_swinney1.json
```

ffmpeg the saved insertions together with frame numbers:

```
ffmpeg -start_number 10000 -framerate 29.97 -y -i $HOME/awecom/data/clips/swinney1/insertion_attempts/homography/swinney1_%06d_ad_insertion.jpg -vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=10000: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5"  -vcodec libx264 -crf 18 -pix_fmt yuv420p -frames 1000 out.mp4
```





