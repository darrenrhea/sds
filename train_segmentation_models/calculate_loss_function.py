from unweighted_l1_loss_per_channel import (
     unweighted_l1_loss_per_channel
)
import torch
import torch.nn.functional as F
from collections import defaultdict
from structure_loss import (
     structure_loss
)

from loss_info import valid_loss_function_family_ids
from awl import AdaptiveWingLoss
from weighted_mse_loss import weighted_mse_loss

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calculate_loss_function(
    loss_function_family_id: str,
    loss_parameters: dict,
    dict_of_output_tensors: dict,
    labels,  # might be multiple labels / targets in the future
    importance_weights,
    metrics: defaultdict  # this is a defaultdict(float) that is mutated by this procedure
):
    """
    There are a variety of families of loss functions that we might want to use.
    Which family is used is determined by the loss_function_family_id.
    Within the family, to get to a specific loss function, These are passed in as loss_parameters.
    """
    assert labels.ndim == 4

    assert loss_function_family_id in valid_loss_function_family_ids

    if loss_function_family_id == "duat":
        P1, P2 = dict_of_output_tensors["P1"], dict_of_output_tensors["P2"]
    else:
        pred = dict_of_output_tensors["outputs"]
    
    assert pred.ndim == 4
    assert pred.size() == labels.size()
    
    if loss_function_family_id == "unweighted_l1":
        # The equality of weighting is in everything:
        # spacially, all regions matter equally.
        # all "goals", like predict floor_not_floor and predict depth_map, matter equally.
        # The direction of the error matters equally, false positives and false negatives are equally bad.
        # When training, the model should not have a sigmoid in it.
        between_zero_and_one = pred.sigmoid()
        
        loss_per_label_channel = unweighted_l1_loss_per_channel(
            input=between_zero_and_one,
            target=labels,
        )
        # all targets weighted equally here:
        loss = torch.sum(loss_per_label_channel)

        for k in range(labels.shape[1]):
            metrics[f"l1_loss_for_target{k}"] += loss_per_label_channel.data.cpu().numpy()[k]

    elif loss_function_family_id == "class_weighted_bce": 
        weight_on_negative = loss_parameters["weight_on_negative"]
        pos_weight = torch.cuda.FloatTensor(
            [[[weight_on_negative]], [[1 - weight_on_negative]]]
        )

        bce = F.binary_cross_entropy_with_logits(
            input=pred,
            target=labels,
            pos_weight=pos_weight
        )
        pred = torch.sigmoid(pred)
        dice = dice_loss(pred, labels)
        loss = bce * bce_weight + dice * (100.0 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * labels.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * labels.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)

    elif loss_function_family_id == "bce_dice_loss":
        bce_weight = loss_parameters["bce_weight"]
        bce = F.binary_cross_entropy_with_logits(pred, labels)
        pred = torch.sigmoid(pred)
        dice = dice_loss(pred, labels)

        loss = bce * bce_weight + dice * (100.0 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * labels.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * labels.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
    
    elif loss_function_family_id == "possiblenewmse":
        loss = weighted_mse_loss(
            input=pred,
            target=labels,
            importance_weights=importance_weights
        )
        # TODO: may need to maintain a separate numerator and denominator for the metrics since all 
        metrics['mse'] += loss.data.cpu().numpy() * labels.size(0)

    elif loss_function_family_id == "mse": 
        # When training, the model should not have a sigmoid in it.
        between_zero_and_one = pred.sigmoid()

        loss = weighted_mse_loss(
            input=between_zero_and_one,
            target=labels,
            importance_weights=importance_weights
        )
        # TODO: may need to maintain a separate numerator and denominator for the metrics since all 
        metrics['mse'] += loss.data.cpu().numpy() * labels.size(0)

    elif loss_function_family_id == "awl":
        wing_loss = AdaptiveWingLoss(whetherWeighted=False)

        loss = wing_loss(pred, labels)

    elif loss_function_family_id == "duat":
        P1 = dict_of_output_tensors["P1"]
        P2 = dict_of_output_tensors["P2"]
        loss_P1 = structure_loss(P1, labels)
        loss_P2 = structure_loss(P2, labels)
        loss = loss_P1 + loss_P2
    else:
        raise Exception(f"Unimplemented loss_function_family_id {loss_function_family_id}")
    

    return loss
