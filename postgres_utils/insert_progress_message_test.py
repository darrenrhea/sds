from insert_run_id import (
     insert_run_id
)
import time
from insert_progress_message import (
     insert_progress_message
)

run_description_jsonable = dict(
    league="nba",
    sport="basketball",
    court_arena_stadium_id="td_garden",
    segmentation_convention="floor_not_floor",
    model_architecture_family_id="u3fasternets",
    original_width=1920,
    original_height=1080,
    patch_width=1920,
    patch_height=1088,
    patch_stride_width=1920,
    patch_stride_height=1088,
    pad_height=8,
)

run_id_uuid = insert_run_id(
    run_description_jsonable=run_description_jsonable
)



for i in range(1, 10):
    loss = 1.0 / i
    progress_message_jsonable = {
        "num_datapoints": i,
        "metrics": {
            "loss": loss
        }
    }
    time.sleep(1)

    insert_progress_message(
        run_id_uuid=run_id_uuid,
        progress_message_jsonable=progress_message_jsonable
    )
