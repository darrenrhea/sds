import sys
import numpy as np
import PIL
import PIL.Image
from pathlib import Path
import numpy as np
import better_json
from colorama import Fore, Style
from FullFrameSegmenter import FullFrameSegmenter
import os
import time
from pathlib import Path
import pprint as pp
import better_json as bj

if False:
    shot_annotations = bj.load("/home/drhea/awecom/data/clips/PHI_CORE_2022-04-16_TOR_PGM/PHI_CORE_2022-04-16_TOR_PGM_shot_annotation_netcams_only.json")

    last_frame_index_in_the_video = 543232
else:
    shot_annotations = bj.load("/home/drhea/awecom/data/clips/PHI_CORE_2022-04-18_TOR_PGM/PHI_CORE_2022-04-18_TOR_PGM_shot_annotation_netcams_only.json")
    # shot_annotations = bj.load("/home/drhea/awecom/data/clips/PHI_CORE_2022-04-18_TOR_PGM/PHI_CORE_2022-04-18_TOR_PGM_shots.json")
    last_frame_index_in_the_video = 523827

wanted_types = ["cam1", "cam1_replay" ,"cam8", "cam8_replay", "cam9", "cam9_replay"]
pp.pprint(shot_annotations)



interval_descriptors = []

an_interval_is_already_started = False
of_what_type = None
the_intervals_first_frame_index = None
for change_event in shot_annotations: #  + [[last_frame_index_in_the_video + 1, "the_end"]]:
    frame_index = change_event[0]
    frame_type = change_event[1]
    if an_interval_is_already_started:
        interval_descriptor = {
            "frame_type": of_what_type,
            "first_frame_index": the_intervals_first_frame_index,
            "last_frame_index": frame_index - 1
        }
        interval_descriptors.append(interval_descriptor)
        # this event 
        an_interval_is_already_started = True
        of_what_type = frame_type
        the_intervals_first_frame_index = frame_index
    else:  # apparently there was nothing before this event:
        an_interval_is_already_started = True
        of_what_type = frame_type
        the_intervals_first_frame_index = frame_index
    
pp.pprint(interval_descriptors)
        
fl = []
for interval_descriptor in interval_descriptors:
    frame_type = interval_descriptor["frame_type"]
    first_frame_index = interval_descriptor["first_frame_index"]
    last_frame_index = interval_descriptor["last_frame_index"]
    if frame_type in wanted_types:
        fl.append([first_frame_index, last_frame_index])

pp.pprint(fl)