# segmentation_utils

We need to choose a video clip, identified by a `clip_id`, such as `swinney1` or `gsw1`.
Basically the clip_ids are the subdirectories of `~/awecom/data/clips/` which include, for example, 
`~/awecom/data/clips/swinney1` and `~/awecom/data/clips/gsw1`.

We will segment each video-frame in the video clip into a 
foreground and background parts.  This is to help with ad insertion.
We really only need the right answer for where the ads are,
and we take advantage of that to process very little of the area of each video frame:
only the area where the camera-parameters for that video-frame suggest the ad is
to be inserted.

The program takes in a config file, see for example, `example_config.json`.

This specifies which model to use,
what video `clip_id` to process/segment into foreground and background,
starting from frame_index `first_frame` and going until `last_frame`.
It needs a `tracking_attempt_id`, to get the camera-parameters for each video frame from
`~/awecom/data/clips/{clip_id}/tracking_attempts/{tracking_attempt_id}/{clip_id}_{frame_index:06d}_camera_parameters.json`.

Similarly but for the output / the segmentation results, the segmentation of each frame will be saved as a black=background and white=foreground png here: 
`~/awecom/data/clips/{clip_id}/masking_attempts/{masking_attempt_id}/{clip_id}_{frame_index:06d}_nonfloor.png`.

We need the locations of the ads so that we can do very little work per frame:

```json
{
    "model_id": "ncaa_kansas_trained_on_6_relevant",
    "clip_id": "swinney1",
    "tracking_attempt_id": "chaz_locked",  // need this for camera_parameters
    "masking_attempt_id": "fast",  // where to save the resulting masks
    "first_frame": 4601,
    "last_frame":  6601,
    "ads": [
        {
            "x_center": -37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "x_center": 37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        }
    ]
}
```

Run it like so:

```
ssh loki
conda activate floor_not_floor
python foreground_segment_video.py example_config.json
```

Use https://github.com/darrenrhea/ad_insertion_video_maker to make an ad insertion:

We need to make another config file, this time for `ad_insertion_video_maker`, like this:

```json
{
    "clip_id": "swinney1",
    "tracking_attempt_id": "chaz_locked",
    "masking_attempt_id": "fast",
    "first_frame": 4601,
    "last_frame":  6601,
    "insertion_attempt_id": "temp",
    "draw_wireframe": true,
    "save_frames": true,
    "ads": [
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        },
        {
            "png_path": "~/awecom/data/ads/cokelogo.png",
            "x_center": 37.6,
            "y_center": 13.2,
            "width": 8,
            "height": 5
        }
    ]
}
```

In particular, `ad_insertion_video_maker` should be using the same `tracking_attempt_id`, i.e. the same camera-solves/camera-parameters as were used by `foreground_segment_video.py`

```
ssh loki
cd ~/r/ad_insertion_video_maker
git switch headless  # the headless branch does not need DISPLAY
./build_project.sh
cd build
bin/ad_insertion ../confs/example_conf.json
```

The resulting insertions are in 

```
~/awecom/data/clips/swinney1/insertion_attempts/<insertion_attempt_id>/
```

so you can see them like:

```
pri ~/awecom/data/clips/swinney1/insertion_attempts/temp/swinney1_006601_ad_insertion.jpg
```

Most people want a video, so use ffmpeg:

```
ffmpeg \
-start_number 4601 \
-framerate 30 \
-y \
-i $HOME/awecom/data/clips/swinney1/insertion_attempts/temp/swinney1_%06d_ad_insertion.jpg \
-vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=4601: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5" \
-vcodec libx264 \
-crf 28 \
-pix_fmt yuv420p \
-frames 2000 \
$HOME/show_n_tell/with_masks.mp4
```

On a local computer/laptop download and watch:

```
mkdir ~/show_n_tell
cd ~/show_n_tell
scp loki:/home/drhea/show_n_tell/with_masks.mp4 .
open with_masks.mp4
```


## Example: the Golden State Warriors


Let's make a config file
called

`gsw1_segmentation_config.json`

specific to the Golden State Warriors,

We need the locations of the ads so that we can do very little work per frame:

```json
{
    "model_id": "gsw1_trained_on_2_relevant",
    "clip_id": "gsw1",
    "tracking_attempt_id": "second",  // need this for camera_parameters
    "masking_attempt_id": "gsw1_trained_on_2_relevant_4cokes",  // where to save the resulting masks
    "first_frame": 150000,
    "last_frame":  151000,
    "ads": [
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": 15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": -15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": 15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": -15,
            "width": 10,
            "height": 6
        }
    ]
}
```

Run it like so:

```
ssh loki
conda activate floor_not_floor
time python foreground_segment_video.py gsw1_segmentation_config.json
```

Use https://github.com/darrenrhea/ad_insertion_video_maker to make an ad insertion:

We need to make another config file, this time for 
ad_insertion by `ad_insertion_video_maker`.
Call it `~/r/ad_insertion_video_maker/confs/gsw1_ad_insertion_config.json`
with this content. 

**Note: the ad placement has to fit within the rectangles defined by the the segmentation config file since the neural network only hit that area.**

```json
{
    "clip_id": "gsw1",
    "tracking_attempt_id": "second",
    "masking_attempt_id": "gsw1_trained_on_2_relevant_4cokes",
    "first_frame": 150000,
    "last_frame":  151000,
    "insertion_attempt_id": "temp",
    "draw_wireframe": false,
    "save_frames": true,
    "ads": [
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": 15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": -37.6,
            "y_center": -15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": 15,
            "width": 10,
            "height": 6
        },
        {
            "png_path": "~/awecom/data/ads/red_cokelogo.png",
            "x_center": 37.6,
            "y_center": -15,
            "width": 10,
            "height": 6
        }
    ]
}
```

In particular, `ad_insertion_video_maker` should be using the same `tracking_attempt_id`, i.e. the same camera-solves/camera-parameters as were used by `foreground_segment_video.py`

```
ssh loki
cd ~/r/ad_insertion_video_maker
git switch headless  # the headless branch does not need DISPLAY
./build_project.sh
cd build
# Because we don't know how to make a directory in C++, you have to make it manually:
mkdir -p /home/drhea/awecom/data/clips/gsw1/insertion_attempts/temp
bin/ad_insertion ../confs/gsw1_ad_insertion_conf.json
```

The resulting insertions are in 

```
~/awecom/data/clips/gsw1/insertion_attempts/<insertion_attempt_id>/
```

so you can see them individually like:

```
pri ~/awecom/data/clips/gsw1/insertion_attempts/temp/gsw1_150000_ad_insertion.jpg
```

Most people want a video, so use ffmpeg:

```
ffmpeg \
-start_number 150000 \
-framerate 60 \
-y \
-i $HOME/awecom/data/clips/gsw1/insertion_attempts/fastai/gsw1_%06d_ad_insertion.jpg \
-vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=150000: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5" \
-vcodec libx264 \
-crf 28 \
-pix_fmt yuv420p \
-frames 700 \
$HOME/show_n_tell/gsw1_with_fastai.mp4

ffmpeg \
-start_number 160000 \
-framerate 60 \
-y \
-i $HOME/awecom/data/clips/gsw1/insertion_attempts/fastai_5/gsw1_%06d_ad_insertion.jpg \
-vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=160000: x=w-tw: y=lh: fontcolor=green: fontsize=50: box=1: boxcolor=black: boxborderw=5" \
-vcodec libx264 \
-crf 18 \
-pix_fmt yuv420p \
$HOME/show_n_tell/gsw1_with_fastai.mp4
```

On a local computer/laptop download and watch:

```
mkdir ~/show_n_tell
cd ~/show_n_tell
scp loki:/home/drhea/show_n_tell/gsw1_with_masks.mp4 .
open gsw1_with_masks.mp4
```
