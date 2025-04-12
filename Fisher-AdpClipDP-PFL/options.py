import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=2, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=4, help="Number of local epochs")### CIFAR-10的 local epochs为4
    parser.add_argument('--global_epoch', type=int, default=40, help="Number of global epochs")### CIFAR-10的global epochs为40
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")### CIFAR-10的batch sizes为64
    parser.add_argument('--num_class', type=int, default=10, help="Label's classes")

    parser.add_argument('--user_sample_rate', type=float, default=0.8, help="Sample rate for user sampling")##可选（随机选择的客户端用于训练的比例）

    parser.add_argument('--target_epsilon', type=float, default=16, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-2, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.5, help="Gradient clipping bound")####初始设置的小一些，因为自适应裁剪的几何更新会使得裁剪阈值指数倍增加变化
    parser.add_argument('--noise_multiplier', type=float, default=0.1)##可选噪声乘子为：{0, 0.01, 0.03, 0.1}

    parser.add_argument('--target_unclipped_quantile', type=float, default=0.5)
    parser.add_argument('--adaptive_clip_learning_rate', type=float, default=0.2)
    
    parser.add_argument('--fisher_threshold', type=float, default=0.5, help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='CIFAR10')

    parser.add_argument('--dir_alpha', type=float, default=10)

    parser.add_argument('--dirStr', type=str, default='')

    parser.add_argument('--store', action='store_true')

    parser.add_argument('--appendix', type=str, default='')



    args = parser.parse_args()
    return args
