from train_a_model import (
     train_a_model
)
from get_datapoint_path_tuples_for_nfl import (
     get_datapoint_path_tuples_for_nfl
)

datapoint_path_tuples = get_datapoint_path_tuples_for_nfl()
other = "thisisshit"
resume_checkpoint_path = None
drop_a_model_this_often = 10
resume_checkpoint_path = "/shared/checkpoints/u3fasternets-floor-136frames-1920x1088-havarti_epoch001500.pt"

train_a_model(
    datapoint_path_tuples=datapoint_path_tuples,
    other=other,
    resume_checkpoint_path=resume_checkpoint_path,
    drop_a_model_this_often=10,
    num_epochs=10,
)
