import torch

from VAE_utils.steplr_scheduler import StepLR
from VAE_utils.multisteplr_scheduler import MultiStepLR
from VAE_utils.consine_lr_scheduler import CosineAnnealingLR

"""
    args.lr_scheduler in 
    ["StepLR", "MultiStepLR", "CosineAnnealingLR"]
    --step-size
    --lr-decay-rate
    --lr-milestones
    --lr-T-max
    --lr-eta-min
"""


def create_scheduler(args, optimizer, **kwargs):
    """
    Create scheduler based on args and the optimizer.
    """
    if args.sched == "no":
        return None
    elif args.sched == "StepLR":
        return StepLR(optimizer, base_lr=args.lr, num_iterations=kwargs.get('num_iterations', 0), 
                      step_size=args.step_size, lr_decay_rate=args.lr_decay_rate)
    elif args.sched == "MultiStepLR":
        return MultiStepLR(optimizer, base_lr=args.lr, num_iterations=kwargs.get('num_iterations', 0), 
                           milestones=args.lr_milestones, lr_decay_rate=args.lr_decay_rate)
    elif args.sched == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, base_lr=args.lr, num_iterations=kwargs.get('num_iterations', 0), 
                                 lr_T_max=args.lr_T_max, lr_eta_min=args.lr_eta_min)
    else:
        raise NotImplementedError(f"Scheduler '{args.sched}' not implemented.")


    return lr_scheduler







