from CameraParameters import (
     CameraParameters
)
from collections import defaultdict
from get_valid_camera_names import (
     get_valid_camera_names
)
from arcpttsd_add_raguls_camera_poses_to_the_segmentation_data import (
     arcpttsd_add_raguls_camera_poses_to_the_segmentation_data
)
import numpy as np
import matplotlib.pyplot as plt

# Get the segmentation data for all nba annotations:
better_segmentation_annotations = (
    arcpttsd_add_raguls_camera_poses_to_the_segmentation_data()
)

camera_name_to_records = defaultdict(list)

for record in better_segmentation_annotations:
    assert isinstance(record, dict)
    court = record["clip_id_info"]["court"]
    assert court is not None, f"record {record} does not have a court key"
    camera_pose = record["camera_pose"]
    camera_name = record["camera_name"]
    if camera_name is not None and camera_pose is not None:
        assert isinstance(camera_pose, dict), f"type(camera_pose) is {type(camera_pose)}"
        camera_name_to_records[camera_name].append(record)
    
valid_camera_names = get_valid_camera_names()

camera_name_mapsto_color = {
    "C01": "red",
    "C02": "green",
    "NETCAM_LEFT": "blue",
    "NETCAM_RIGHT": "purple",
    "SPIDER": "orange",
}

xs = []
ys = []
colors = []
for camera_name in valid_camera_names:
    records = camera_name_to_records[camera_name]
    for record in records:
        camera_pose = record["camera_pose"]
        x, y, z = camera_pose["loc"]
        xs.append(x)
        ys.append(y)
        colors.append(camera_name_mapsto_color[camera_name])


# Sample x, y coordinates
# x = np.random.rand(50) * 10  # 50 random x values between 0 and 10
# y = np.random.rand(50) * 10  # 50 random y values between 0 and 10
# colors = np.random.rand(50, 3)  # Generate random RGB colors

plt.figure(figsize=(8, 6))
plt.scatter(xs, ys, c=colors, s=100, edgecolors='black', alpha=0.75)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Colorful Scatter Plot of X, Y Coordinates')
plt.grid(True)
plt.show()