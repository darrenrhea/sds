import torch


def load_checkpoint(
    in_path,
    model = None,
    optimizer = None,
    scheduler = None,
    multigpu = False
):
    """
    TODO: get model_architecture_id, patch_size etc from checkpoint
    or from some JSON thing.
    """
    checkpoint = torch.load(in_path, map_location=torch.device('cpu'))

    if multigpu:
        # TODO: shouldnt this call get_model_state_dict?
        from collections import OrderedDict
        model_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            if k[:7] == 'module.':
                k = k[7:]
            model_state_dict[k] = v
        checkpoint['model'] = model_state_dict

    checkpoint['model'] = model_state_dict

    if model:
        #print('keys', list(checkpoint['model'].keys()))
        model.load_state_dict(checkpoint['model'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler and 'lr_scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])

    return checkpoint

