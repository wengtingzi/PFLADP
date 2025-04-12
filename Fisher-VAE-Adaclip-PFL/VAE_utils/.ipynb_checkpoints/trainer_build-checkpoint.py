import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from VAE_utils.normal_trainer import NormalTrainer
from VAE_utils.optim_build import create_optimizer
from VAE_utils.loss_build import create_loss
from VAE_utils.lr_build import create_scheduler



def create_trainer(args, device, model, **kwargs):
    """
    Simplified version of create_trainer, now only needs minimal params.
    """

    # 使用args传递其他需要的参数
    optimizer = create_optimizer(args, model, **kwargs)
    criterion = create_loss(args, device, **kwargs)
    lr_scheduler = create_scheduler(args, optimizer, **kwargs)   # no for FedAvg

    model_trainer = NormalTrainer(model, device, criterion, optimizer, lr_scheduler, args, **kwargs)

    return model_trainer












