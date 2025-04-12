import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning with VAE")

    # --------------------------------基本设置--------------------------------
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--client_num_in_total', type=int, default=10)
    parser.add_argument('--client_num_per_round', type=int, default=5)
    parser.add_argument('--client_select', type=str, default='random')
    parser.add_argument('--local_epoch', type=int, default=4, help="Number of local epochs")### CIFAR-10的 local epochs为4
    parser.add_argument('--global_epoch', type=int, default=40, help="Number of global epochs")### CIFAR-10的global epochs为40
    parser.add_argument('--global_epochs_per_round', type=int, default=1)
    parser.add_argument('--comm_round', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--fisher_threshold', type=float, default=0.4, help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")
    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--dir_alpha', type=float, default=100)
    parser.add_argument('--dirStr', type=str, default='')

    # --------------------------------优化器与调度--------------------------------
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=False)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--client_optimizer', type=str, default='no')
    parser.add_argument('--server_optimizer', type=str, default='no')
    parser.add_argument('--sched', type=str, default='no')
    parser.add_argument('--lr_decay_rate', type=float, default=0.992)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[30, 60])
    parser.add_argument('--lr_T_max', type=int, default=10)
    parser.add_argument('--lr_eta_min', type=float, default=0)
    parser.add_argument('--lr_warmup_type', type=str, default='constant')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--lr_warmup_value', type=float, default=0.1)

    # --------------------------------数据设置--------------------------------
    parser.add_argument('--data_dir', type=str, default='./../data')
    parser.add_argument('--partition_method', type=str, default='hetero')
    parser.add_argument('--partition_alpha', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")### CIFAR-10的batch sizes为64
    parser.add_argument('--user_sample_rate', type=float, default=0.8)####可选（随机选择的客户端用于训练的比例）
    parser.add_argument('--data_sampler', type=str, default='random')
    parser.add_argument('--dataset_aug', type=str, default='default')
    parser.add_argument('--dataset_resize', type=bool, default=False)
    parser.add_argument('--dataset_load_image_size', type=int, default=32)
    parser.add_argument('--data_efficient_load', type=bool, default=True)
    parser.add_argument('--dirichlet_min_p', type=float, default=None)
    parser.add_argument('--dirichlet_balance', type=bool, default=False)
    parser.add_argument('--data_load_num_workers', type=int, default=1)
    parser.add_argument('--TwoCropTransform', type=bool, default=False)

    # --------------------------------模型设置--------------------------------
    parser.add_argument('--model', type=str, default='resnet18_v2')
    parser.add_argument('--model_input_channels', type=int, default=3)
    parser.add_argument('--model_output_dim', type=int, default=10)
    parser.add_argument('--model_out_feature', type=bool, default=False)
    parser.add_argument('--model_out_feature_layer', type=str, default='last')
    parser.add_argument('--model_feature_dim', type=int, default=512)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_dir', type=str, default=' ')
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--algorithm', type=str, default='FedAvg')

    # --------------------------------FedProx / Scaffold--------------------------------
    parser.add_argument('--fedprox', type=bool, default=False)
    parser.add_argument('--fedprox_mu', type=float, default=0.1)
    parser.add_argument('--scaffold', type=bool, default=False)

    # --------------------------------VAE 设置--------------------------------
    parser.add_argument('--VAE', type=bool, default=True)
    parser.add_argument('--VAE_local_epoch', type=int, default=1)
    parser.add_argument('--VAE_d', type=int, default=32)
    parser.add_argument('--VAE_z', type=int, default=2048)
    parser.add_argument('--VAE_batch_size', type=int, default=64)
    parser.add_argument('--VAE_aug_batch_size', type=int, default=64)
    parser.add_argument('--VAE_comm_round', type=int, default=1)
    parser.add_argument('--VAE_client_num', type=int, default=10)
    parser.add_argument('--VAE_re', type=float, default=5.0)
    parser.add_argument('--VAE_ce', type=float, default=2.0)
    parser.add_argument('--VAE_kl', type=float, default=0.005)
    parser.add_argument('--VAE_x_ce', type=float, default=0.4)
    parser.add_argument('--VAE_std1', type=float, default=0.2)
    parser.add_argument('--VAE_std2', type=float, default=0.25)
    parser.add_argument('--VAE_adaptive', type=bool, default=True)
    parser.add_argument('--VAE_sched', type=str, default='cosine')
    parser.add_argument('--VAE_sched_lr_ate_min', type=float, default=2e-3)
    parser.add_argument('--VAE_step', type=str, default='+')
    parser.add_argument('--VAE_mixupdata', type=bool, default=False)
    parser.add_argument('--VAE_curriculum', type=bool, default=True)
    parser.add_argument('--VAE_mean', type=int, default=0)
    parser.add_argument('--VAE_alpha', type=float, default=2.0)

    # --------------------------------差分隐私设置--------------------------------
    parser.add_argument('--target_epsilon', type=float, default=16, help="Target privacy budget epsilon")
    parser.add_argument('--noise_type', type=str, default='Gaussian', help='Type of noise used in VAE')

    parser.add_argument('--target_delta', type=float, default=1e-2, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.1, help="Gradient clipping bound")####初始设置的小一些，因为自适应裁剪的几何更新会使得裁剪阈值指数倍增加变化
    parser.add_argument('--noise_multiplier', type=float, default=0.1)##可选噪声乘子为：{0, 0.01, 0.03, 0.1}
    
    parser.add_argument('--target_unclipped_quantile', type=float, default=0.5)
    parser.add_argument('--adaptive_clip_learning_rate', type=float, default=0.2)

    # --------------------------------任务设置与运行模式--------------------------------
    parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--mode', type=str, default='standalone')
    parser.add_argument('--instantiate_all', type=bool, default=True)
    parser.add_argument('--client_index', type=int, default=0)
    parser.add_argument('--exchange_model', type=bool, default=True)
    parser.add_argument('--loss_fn', type=str, default='CrossEntropy')
    parser.add_argument('--max_epochs', type=int, default=90)

    # --------------------------------记录设置--------------------------------
    parser.add_argument('--record_tool', type=str, default='wandb')
    parser.add_argument('--wandb_record', type=bool, default=False)
    parser.add_argument('--store', action='store_true')
    parser.add_argument('--appendix', type=str, default='')
    parser.add_argument('--level', type=str, default='INFO')
    parser.add_argument('--test', type=bool, default=True)

    args = parser.parse_args()
    return args
