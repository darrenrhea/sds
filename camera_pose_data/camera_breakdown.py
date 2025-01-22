import json
from pathlib import Path

# Path to your JSONL file
in_path = "20241017HoustonRockets_out.jsonl"
# out_path = "20241017HoustonRockets_netcam_frames.jsonl"
out_list_path = Path('~/r/frame_attributes/hou-sas-2024-10-17-sdi_netcams.json5').expanduser()
# in_path = "out_NEWvsHOU_20241103.jsonl"
# out_path = "NEWvsHOU_20241103_netcam_frames.jsonl"
# out_list_path = Path('~/r/frame_attributes/hou-gsw-2024-11-02-sdi_netcams.json5').expanduser()
# in_path = "out_NEWvsHOU_20241112.jsonl"
# out_path = "NEWvsHOU_20241112_netcam_frames.jsonl"
# out_list_path = Path('~/r/frame_attributes/hou-was-2024-11-11-sdi_netcams.json5').expanduser()
out_data = []
out_list = []

# Read the file line by line
with open(in_path, 'r') as file:
    data = [json.loads(line) for line in file]

# Print the parsed data
netcam_counter = 0
for entry in data:
    camera_name = entry['name']
    if 'NETCAM' in camera_name:
        frame_number = entry['frame_number']
        # print(f"{type(frame_number)}=") 
        # print(f"{isinstance(frame_number, int)}=")       
        # print(camera_name)
        netcam_counter += 1
        out_data.append({"frame_number":frame_number, "name":camera_name})
        out_list.append(frame_number)

print(f"{entry.keys()=}")
print(f"number of netcam frames: {netcam_counter}")

# Write data to JSONL file
# with open(out_path, 'w') as file:
#     for item in out_data:
#         json_line = json.dumps(item)  # Convert dictionary to JSON string
#         file.write(json_line + '\n')  # Write each JSON string as a new line

with open(out_list_path, 'w') as file:
    json.dump(out_list, file, indent=4)