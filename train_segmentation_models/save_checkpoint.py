import torch

def save_checkpoint(
    model,
    optimizer,
    epoch,
    out_path,
    scheduler = None,
    multigpu = False,
    extra = None
):
    checkpoint = {
        'model': model.module.state_dict() if multigpu else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if scheduler:
        checkpoint['lr_scheduler'] = scheduler.state_dict()
    if extra:
        checkpoint['extra'] = extra
    torch.save(checkpoint, out_path)
