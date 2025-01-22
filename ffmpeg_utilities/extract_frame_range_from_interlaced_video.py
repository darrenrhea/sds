# We needed frames 91000 to 121000 from the video EB_23-24_R21_PAR-MTA.mxf.  We used the following command to extract them,
# based on the thought in extract_single_frame_from_interlaced_video.py
"""
time ffmpeg \
-y \
-nostdin \
-accurate_seek \
-ss \
1819.955 \
-i \
/Volumes/NBA/Euroleague/EB_23-24_R21_PAR-MTA.mxf \
-vf \
yadif=mode=1 \
-f \
image2 \
-vsync \
0 \
-pix_fmt \
yuvj422p \
-q:v \
2 \
-qmin \
2 \
-qmax \
2 \
-frames:v 30010 \
-loglevel error \
-start_number 90998 \
/Users/awecom/a/clips/belgrade2024-01-12-1080i-yadif/frames/belgrade2024-01-12-1080i-yadif_%06d_original.jpg
"""