# video_frame_data_utils

This repo intends to allow you to grab any video frame from any of our videos,
regardless of what computer you are on.

Additionally, it installs a CLI tool, priframe, that allows you to look at any video frame in any video:

```bash
priframe brewcub $(( 23077 + 112 ))
```

This is a private Python library that you should install like this:

```bash
cd ~/r
git clone git@github.com:darrenrhea/video_frame_data_utils
cd ~/r/video_frame_data_utils
pip install -e . --no-deps

# Try it:
priframe brewcub 23094
```


