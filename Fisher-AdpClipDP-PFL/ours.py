import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from options import parse_args
from data import *
from net import *
from tqdm import tqdm
from utils import compute_fisher_diag
from tqdm.auto import trange, tqdm
import copy
import sys
import random
from torch.optim import Optimizer
import datetime
import tensorflow as tf
import sys
import tensorflow_privacy as tfp
import numpy as np
from tensorflow_privacy.privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipSumQuery
from tensorflow_privacy.privacy.dp_query.normalized_query import NormalizedQuery
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from typing import Any, NamedTuple, Optional

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = args.dataset
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {dataset} '
        f'--num_clients {num_clients} '
        f'--local_epoch {local_epoch} '
        f'--global_epoch {global_epoch} '
        f'--batch_size {batch_size} '
        f'--target_epsilon {target_epsilon} '
        f'--target_delta {target_delta} '
        f'--clipping_bound {clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.dir_alpha}'
        f'.txt'
        ,'a'
        )
    sys.stdout = file

def tf_tensor_to_torch(tensors):
    if isinstance(tensors, list):
        return [torch.tensor(t.numpy()) for t in tensors]
    else:
        return torch.tensor(tensors.numpy())

def torch_tensor_to_tf(tensors):
    if isinstance(tensors, list):
        return [tf.convert_to_tensor(t.detach().cpu().numpy(), dtype=tf.float32) for t in tensors]
    else:
        return tf.convert_to_tensor(tensors.detach().cpu().numpy(), dtype=tf.float32)
        
def get_epsilon(
    num_examples,
    batch_size,
    noise_multiplier,
    epochs,
    delta=1e-5
) -> float:
    """返回当前训练设置下的 epsilon 值"""
    statement = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(
        num_examples, # 总的训练样本数
        batch_size, # 客户端训练样本批次大小
        epochs, # 客户端训练次数大小
        noise_multiplier,
        delta, 
        True,  # used_microbatching
        None,  # max_examples_per_user
        compute_dp_sgd_privacy_lib.AccountantType.RDP
    )
    
    # 从打印语句中提取 epsilon
    for line in statement.splitlines():
        if "Epsilon with each example occurring once per epoch" in line:
            epsilon = float(line.split(":")[1])
            return epsilon
    raise ValueError("无法提取 epsilon 值")

def adaptive_clip_noise_params(
    noise_multiplier: float,
    expected_clients_per_round: float,
    clipped_count_stddev: Optional[float] = None,
) -> tuple[float, float]:
  if noise_multiplier > 0.0:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.05 * expected_clients_per_round

    if noise_multiplier >= 2 * clipped_count_stddev:
      raise ValueError(
          f'clipped_count_stddev = {clipped_count_stddev} (defaults to '
          '0.05 * `expected_clients_per_round` if not specified) is too low '
          'to achieve the desired effective `noise_multiplier` '
          f'({noise_multiplier}). You must either increase '
          '`clipped_count_stddev` or decrease `noise_multiplier`.'
      )

    value_noise_multiplier = (
        noise_multiplier**-2 - (2 * clipped_count_stddev) ** -2
    ) ** -0.5

    added_noise_factor = value_noise_multiplier / noise_multiplier
    if added_noise_factor >= 2:
      warnings.warn(
          f'A significant amount of noise ({added_noise_factor:.2f}x) has to '
          'be added for record aggregation to achieve the desired effective '
          f'`noise_multiplier` ({noise_multiplier}). If you are manually '
          'specifying `clipped_count_stddev` you may want to increase it. Or '
          'you may need more `expected_clients_per_round`.'
      )
  else:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.0
    value_noise_multiplier = 0.0

  return value_noise_multiplier, clipped_count_stddev

def local_update(model, dataloader, global_model):


    fisher_threshold = args.fisher_threshold
    model = model.to(device)
    global_model = global_model.to(device)

    w_glob = [param.clone().detach() for param in global_model.parameters()]

    fisher_diag = compute_fisher_diag(model, dataloader)


    u_loc, v_loc = [], []
    for param, fisher_value in zip(model.parameters(), fisher_diag):
        u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
        u_loc.append(u_param)
        v_loc.append(v_param)

    u_glob, v_glob = [], []
    for global_param, fisher_value in zip(global_model.parameters(), fisher_diag):
        u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
        v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
        u_glob.append(u_param)
        v_glob.append(v_param)

    for u_param, v_param, model_param in zip(u_loc, v_glob, model.parameters()):
        model_param.data = u_param + v_param

    saved_u_loc = [u.clone() for u in u_loc]

    def custom_loss(outputs, labels, param_diffs, reg_type):
        ce_loss = F.cross_entropy(outputs, labels)
        if reg_type == "R1":
            reg_loss = (args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

        elif reg_type == "R2":
            C = args.clipping_bound
            norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
            reg_loss = (args.lambda_2 / 2) * torch.norm(norm_diff - C)

        else:
            raise ValueError("Invalid regularization type")

        return ce_loss + reg_loss
    

    optimizer1 = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer1.zero_grad()
            outputs = model(data)
            param_diffs = [u_new - u_old for u_new, u_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R1")
            loss.backward()
            with torch.no_grad():
                for model_param, u_param in zip(model.parameters(), u_loc):
                    model_param.grad *= (u_param != 0)
            optimizer1.step()
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.local_epoch):
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            optimizer2.zero_grad()
            outputs = model(data)
            param_diffs = [model_param - w_old for model_param, w_old in zip(model.parameters(), w_glob)]
            loss = custom_loss(outputs, labels, param_diffs, "R2")
            loss.backward()
            with torch.no_grad():
                for model_param, v_param in zip(model.parameters(), v_glob):
                    model_param.grad *= (v_param != 0)
            optimizer2.step()

    with torch.no_grad():
        update = [(new_param - old_param).clone() for new_param, old_param in zip(model.parameters(), w_glob)]


    model = model.to('cpu')
    return update

def test(client_model, client_testloader):
    client_model.eval()
    client_model = client_model.to(device)

    num_data = 0


    correct = 0
    with torch.no_grad():
        for data, labels in client_testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = client_model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            num_data += labels.size(0)
    
    accuracy = 100.0 * correct / num_data

    client_model = client_model.to('cpu')

    return accuracy

###--------主函数-------####
def main():

    mean_acc_s = []
    acc_matrix = []
    if dataset == 'MNIST':

        train_dataset, test_dataset = get_mnist_datasets()
        clients_train_set = get_clients_datasets(train_dataset, num_clients)
        client_data_sizes = [len(client_dataset) for client_dataset in clients_train_set]
        clients_train_loaders = [DataLoader(client_dataset, batch_size=batch_size) for client_dataset in clients_train_set]
        clients_test_loaders = [DataLoader(test_dataset) for i in range(num_clients)]

        clients_models = [mnistNet() for _ in range(num_clients)]
        global_model = mnistNet()
    elif dataset == 'CIFAR10':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_CIFAR10(args.dir_alpha, num_clients)

        clients_models = [cifar10Net() for _ in range(num_clients)]
        global_model = cifar10Net()
    elif dataset == 'FEMNIST':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)

        clients_models = [femnistNet() for _ in range(num_clients)]
        global_model = femnistNet()
    elif dataset == 'SVHN':
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_SVHN(args.dir_alpha, num_clients)

        clients_models = [SVHNNet() for _ in range(num_clients)]
        global_model = SVHNNet()
    else:
        print('undifined dataset')
        assert 1==0
    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())
    clip=args.clipping_bound
    value_noise_multiplier = None
    clipped_count_stddev = None
    for epoch in trange(global_epoch):
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        clients_model_updates = []
        clients_accuracies = []
        
        ####------- client Train-------#####
        for idx, (client_model, client_trainloader, client_testloader) in enumerate(zip(sampled_clients_models, sampled_clients_train_loaders, sampled_clients_test_loaders)):
            if not args.store:
                tqdm.write(f'client:{idx+1}/{args.num_clients}')
            client_update = local_update(client_model, client_trainloader, global_model)
            clients_model_updates.append(client_update)
            accuracy = test(client_model, client_testloader)
            clients_accuracies.append(accuracy)
        print(clients_accuracies)
        mean_acc_s.append(sum(clients_accuracies)/len(clients_accuracies))
        acc_matrix.append(clients_accuracies)
         ####------- Server Aggregator-------#####
        sample_client_num=len(sampled_client_indices)
        N = sum([client_data_sizes[i] for i in sampled_client_indices])
        if epoch==0:##-------如果是第一轮聚合-------##
            # 初始化自适应裁剪查询
            value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
                noise_multiplier=args.noise_multiplier,
                expected_clients_per_round=sample_client_num,
                clipped_count_stddev=None)
            query = QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=clip,
                noise_multiplier=value_noise_multiplier,
                target_unclipped_quantile=args.target_unclipped_quantile,
                learning_rate=args.adaptive_clip_learning_rate,
                clipped_count_stddev=clipped_count_stddev,
                expected_num_records=sample_client_num,
                geometric_update=True)
            # 用NormalizedQuery包裹，实现平均值聚合
            query = NormalizedQuery(query, denominator=sample_client_num)
            # 初始化全局状态（第一次聚合时）
            global_state = query.initial_global_state()
            sample_params = query.derive_sample_params(global_state) # 获取采样参数
            # 初始化样本状态（累加器）
            example_update = torch_tensor_to_tf(clients_model_updates[0])  # 获取一个样例更新
            sample_state = query.initial_sample_state(example_update)  # 使用实际结构生成状态
            tf_clients_model_updates = []
            for update in clients_model_updates:
                tf_update = torch_tensor_to_tf(update)  # update 是 List[Tensor]
                tf_clients_model_updates.append(tf_update)
            # 预处理并累加所有客户端更新
            for update in tf_clients_model_updates:
                record = query.preprocess_record(sample_params, update)
                sample_state = query.accumulate_preprocessed_record(sample_state, record)
            # 聚合并加噪输出
            result, new_global_state, _ = query.get_noised_result(sample_state, global_state)
            clip = new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()
            print("\n更新后的裁剪范数:", clip)
            pytorch_update = [tf_tensor_to_torch(t) for t in result]
            # Apply the aggregated updates to the global model parameters
            with torch.no_grad():
                for global_param, update in zip(global_model.parameters(), pytorch_update):
                    global_param.add_(update)
            print('Z_delta是：')    
            print(value_noise_multiplier)
            print('标准差是：')    
            print(clipped_count_stddev)
            epsilon = get_epsilon(
                num_examples=N,
                batch_size=batch_size,
                noise_multiplier=value_noise_multiplier,
                epochs=args.local_epoch,
                delta=args.target_delta
            )
            print(f"第{epoch}轮累计使用到的隐私预算为 ε ≈ {epsilon:.4f}")
            if epsilon >= args.target_epsilon:
                print(f"已达到隐私预算 ε={epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break
        
        else:##-------第二轮及之后的聚合-------##
            # 初始化自适应裁剪查询
            print(value_noise_multiplier)
            print(clipped_count_stddev)
            query = QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=clip,
                noise_multiplier=value_noise_multiplier,
                target_unclipped_quantile=args.target_unclipped_quantile,
                learning_rate=args.adaptive_clip_learning_rate,
                clipped_count_stddev=clipped_count_stddev,
                expected_num_records=sample_client_num,
                geometric_update=True)
            # 用NormalizedQuery包裹，实现平均值聚合
            query = NormalizedQuery(query, denominator=sample_client_num)
            # 设置全局状态（第一次聚合时）
            global_state = new_global_state
            sample_params = query.derive_sample_params(global_state) # 获取采样参数
            # 初始化样本状态（累加器）
            example_update = torch_tensor_to_tf(clients_model_updates[0])  # 获取一个样例更新
            sample_state = query.initial_sample_state(example_update)  # 使用实际结构生成状态
            tf_clients_model_updates = []
            for update in clients_model_updates:
                tf_update = torch_tensor_to_tf(update)  # update 是 List[Tensor]
                tf_clients_model_updates.append(tf_update)
            # 预处理并累加所有客户端更新
            for update in tf_clients_model_updates:
                record = query.preprocess_record(sample_params, update)
                sample_state = query.accumulate_preprocessed_record(sample_state, record)
            # 聚合并加噪输出
            result, new_global_state, _ = query.get_noised_result(sample_state, global_state)
            clip = new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()
            print("\n更新后的裁剪范数:", clip)
            pytorch_update = [tf_tensor_to_torch(t) for t in result]
            with torch.no_grad():
                for global_param, update in zip(global_model.parameters(), pytorch_update):
                    global_param.add_(update)
            epsilon = get_epsilon(
                num_examples=N,
                batch_size=batch_size,
                noise_multiplier=value_noise_multiplier,
                epochs=args.local_epoch,
                delta=args.target_delta
            )
            print(f"第{epoch}轮累计使用到的隐私预算为 ε ≈ {epsilon:.4f}")
            if epsilon >= args.target_epsilon:
                print(f"已达到隐私预算 ε={epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break
                
    char_set = '1234567890abcdefghijklmnopqrstuvwxyz'
    ID = ''
    for ch in random.sample(char_set, 5):
        ID = f'{ID}{ch}'
    print(
        f'===============================================================\n'
        f'task_ID : '
        f'{ID}\n'
        f'main_yxy\n'
        f'mean accuracy : \n'
        f'{mean_acc_s}\n'
        f'acc matrix : \n'
        f'{torch.tensor(acc_matrix)}\n'
        f'===============================================================\n'
    )


if __name__ == '__main__':
    main()

