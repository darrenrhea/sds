from train_a_model import (
     train_a_model
)
from get_datapoint_path_tuples_for_rockets_core import (
     get_datapoint_path_tuples_for_rockets_core
)

datapoint_path_tuples = get_datapoint_path_tuples_for_rockets_core()
other = "darrenjerry"
# resume_checkpoint_path = None
resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-6551frames-1920x1088-darrenjerry_epoch000161.pt"

train_a_model(
    datapoint_path_tuples=datapoint_path_tuples,
    other=other,
    resume_checkpoint_path=resume_checkpoint_path,
)
