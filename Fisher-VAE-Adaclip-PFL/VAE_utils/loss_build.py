import torch.nn as nn



def create_loss(args, device=None, **kwargs):
    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function '{args.loss_fn}' not implemented.")

    return loss_fn
















