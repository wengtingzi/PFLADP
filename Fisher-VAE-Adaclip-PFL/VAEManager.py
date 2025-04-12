import torch
from net import *
import random
from VAE_utils.FL_VAE import *
from VAE_utils.trainer_build import create_trainer
from VAE_utils.model_build import create_model
from VAE_utils.Dataset_3Types_ImageData import Dataset_3Types_ImageData
from VAE_utils.randaugment4fixmatch import RandAugmentMC
from VAE_utils.data_utils import average_named_params
from VAE_utils.set import *
from VAE_utils.AdamW import AdamW
from VAE_utils.tool import *
from copy import deepcopy
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

class VAEManager:
    def __init__(self, args,train_dataloader,test_dataloader, device):
        self.client_list = []
        self.aggregator = None
        self.args = args
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device=device
        self.local_share_data1=None
        self.local_share_data2=None
        self.global_share_dataset1 = None
        self.global_share_dataset2 = None
        self.global_share_data_y = None
    
    def setup_vae_client_list(self):
        for client_index in range(self.args.num_clients):
            if self.args.VAE:
                VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device)

            model_trainer = create_trainer(self.args, self.device, model=model,client_index=client_index, role='client')
            client = Client_VAE(train_dataloader=self.train_dataloader[client_index],
                             test_dataloader=self.test_dataloader[client_index],
                             device=self.device, args=self.args, model_trainer=model_trainer,
                             vae_model=VAE_model)

            self.client_list.append(client)
        return self.client_list
    
    def setup_vae_server(self):
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                            device=self.device)
        if self.args.VAE:
            VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

        model_trainer = create_trainer(self.args, self.device,model=model,server_index=0, role='server')

        self.aggregator = Aggregator_VAE(train_dataloader=self.train_dataloader,test_dataloader=self.test_dataloader,
                                         device=self.device, args=self.args, model_trainer=model_trainer,vae_model=VAE_model)
    
    ## VAE训练最主要的程序
    def share_data_step(self):
        for round in range(self.args.VAE_comm_round):
            client_indexes = self.client_sample_for_VAE(round, self.args.num_clients, self.args.VAE_client_num)
            for client_index in client_indexes:
                client = self.client_list[client_index]
                client.train_vae_model(round)

            self.aggregate_sampled_client_vae(client_indexes)
            # self.aggregator.test_on_server_by_vae(round)

        for client in self.client_list:
            client.generate_data_by_vae()
            if self.global_share_dataset1 is None:
                self.global_share_dataset1 = client.local_share_data1
                self.global_share_dataset2 = client.local_share_data2
                self.global_share_data_y = client.local_share_data_y
            else:
                self.global_share_dataset1 = torch.cat((self.global_share_dataset1, client.local_share_data1))
                self.global_share_dataset2 = torch.cat((self.global_share_dataset2, client.local_share_data2))
                self.global_share_data_y = torch.cat((self.global_share_data_y, client.local_share_data_y))
        return self.global_share_dataset1,self.global_share_dataset2,self.global_share_data_y

    ## share_data_step附属函数
    def client_sample_for_VAE(self, round, num_clients, VAE_client_num):
        if num_clients == VAE_client_num:
            client_indexes = [client_index for client_index in range(num_clients)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            np.random.seed(self.args.VAE_comm_round - round)
            if self.args.client_select == "random":
                num = min(VAE_client_num, num_clients)
                client_indexes = np.random.choice(range(num_clients), num, replace=False)

        logging.info("VAE sampling client_indexes = %s" % str(client_indexes))
        return client_indexes
    ## share_data_step附属函数
    def aggregate_sampled_client_vae(self,client_indexes):
        model_list = []
        training_data_num = 0
        data_num_list = []
        aggregate_weight_list = []
        for client_index in client_indexes:
            client = self.client_list[client_index]
            model_list.append((client.local_sample_number, client.get_vae_para()))
            data_num_list.append(client.local_sample_number)
            training_data_num += client.local_sample_number
        for i in range(0, len(data_num_list)):
            local_sample_number = data_num_list[i]
            weight_by_sample_num = local_sample_number / training_data_num
            aggregate_weight_list.append(weight_by_sample_num)

        averaged_vae_params = average_named_params(
            model_list,  # from sampled client model_list  [(sample_number, model_params)]
            aggregate_weight_list
        )
        self.aggregator.set_vae_param(averaged_vae_params)
        for client in self.client_list:
            client.set_vae_para(averaged_vae_params)


class Client_VAE:
    def __init__(self,train_dataloader,test_dataloader,device, args, model_trainer, vae_model):

        self.trainer = model_trainer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.args= args
        self.vae_model = vae_model
        self.vae_optimizer =  AdamW([
            {'params': self.vae_model.parameters()}
        ], lr=1.e-3, betas=(0.9, 0.999), weight_decay=1.e-6)
        self.local_sample_number = len(self.train_dataloader.dataset) 

    ## get_global_share_data附属函数
    def construct_mix_dataloader(self, share_data1, share_data2, share_y):
        # two dataloader including shared data from server and local original dataloader
        train_ori_transform = transforms.Compose([])
        if self.args.dataset == 'femnist':
            train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['femnist']:
            train_ori_transform.transforms.append(RandAugmentMC(n=3, m=10))
        train_ori_transform.transforms.append(transforms.ToTensor())

        train_share_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
        
        epoch_data1, epoch_label1 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=1)
        epoch_data2, epoch_label2 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=2)
        
        train_dataset = Dataset_3Types_ImageData(
            self.train_dataloader.dataset.data, 
            epoch_data1, epoch_data2,
            self.train_dataloader.dataset.targets, 
            epoch_label1, epoch_label2,
            transform=train_ori_transform,
            share_transform=train_share_transform
        )

        self.local_train_mixed_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=False
        )
        return self.local_train_mixed_dataloader 

    ## construct_mix_dataloader附属函数
    def sample_iid_data_from_share_dataset(self, share_data1, share_data2, share_y, share_data_mode=1):
        random.seed(random.randint(0, 10000))

        if share_data_mode == 1 and share_data1 is None:
            raise RuntimeError("Not get shared data TYPE1")
        if share_data_mode == 2 and share_data2 is None:
            raise RuntimeError("Not get shared data TYPE2")

        sample_num = self.local_sample_number
        sample_num_each_cls = sample_num // self.args.num_classes
        last = sample_num - sample_num_each_cls * self.args.num_classes

        np_y = np.array(share_y.cpu())
        selected_indexes = []

        for label in range(self.args.num_classes):
            indexes = list(np.where(np_y == label)[0])
            if len(indexes) >= sample_num_each_cls:
                sample = random.sample(indexes, sample_num_each_cls)
            else:
                sample = indexes  # 全部拿进来
                print(f"[Info] 类别 {label} 样本不足（{len(indexes)}），全部采样。")
            selected_indexes.extend(sample)

        # 如果最后还有剩余 quota，就随机采一些补上
        if last > 0:
            all_indexes = list(range(len(share_y)))
            remain_indexes = list(set(all_indexes) - set(selected_indexes))
            if len(remain_indexes) >= last:
                last_sample = random.sample(remain_indexes, last)
            else:
                last_sample = remain_indexes  # 不够也没办法
                print(f"[Info] 剩余样本不足用于补全最后 {last} 个样本，只采了 {len(remain_indexes)} 个。")
            selected_indexes.extend(last_sample)

        selected_indexes = torch.tensor(selected_indexes)

        if share_data_mode == 1:
            epoch_data = share_data1[selected_indexes]
        elif share_data_mode == 2:
            epoch_data = share_data2[selected_indexes]
        epoch_label = share_y[selected_indexes]

        return epoch_data, epoch_label

    
    ## share_data_step附属函数
    def train_vae_model(self, round):
        train_transform = transforms.Compose([])
        if self.args.dataset == 'femnist':
            train_transform.transforms.append(transforms.Resize(32))

        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['femnist']:
            train_transform.transforms.append(RandAugmentMC(n=3, m=10))
        train_transform.transforms.append(transforms.ToTensor())
        
        train_dataloader = self.train_dataloader
        self.vae_model.to(self.device)
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + self.args.VAE_local_epoch):
            self.classifier_train(self.vae_optimizer, train_dataloader)
            self.VAE_train(self.vae_optimizer, train_dataloader)
            self.train_whole_process(round,epoch, self.vae_optimizer, train_dataloader)
        self.vae_model.cpu()
    ## train_vae_model附属函数
    def classifier_train(self, optimizer, trainloader):
        self.vae_model.train()
        for name, parameter in self.vae_model.named_parameters():
            if 'classifier' not in name:
                parameter.requires_grad = False
            else: 
                parameter.requires_grad = True

        for x, y in trainloader:
            x, y = x.to(self.device), y.to(self.device).view(-1,)
            optimizer.zero_grad() 
            out = self.vae_model.get_classifier()(x)
            loss = F.cross_entropy(out, y)
            loss.backward() 
            optimizer.step() 
    ## train_vae_model附属函数
    def VAE_train(self, optimizer, trainloader):
        self.vae_model.train()
        self.vae_model.requires_grad_(True)
        for x,y in trainloader: 
            batch_size = x.size(0)
            if batch_size < 4:  
                continue
            x = x.to(self.device)  
            optimizer.zero_grad()  
            _, _, gx, mu, logvar, _, _, _ = self.vae_model(x)
            l1 = F.mse_loss(gx, x)
            l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l3 /= (self.args.VAE_batch_size * 3 * self.args.VAE_z)

            loss = self.args.VAE_re * l1 + self.args.VAE_kl * l3
            loss.backward()
            optimizer.step()
    ## train_vae_model附属函数
    def train_whole_process(self, round, epoch, optimizer, trainloader):
        self.vae_model.train()
        self.vae_model.training = True
        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_ce = AverageMeter()
        loss_kl = AverageMeter()
        top1 = AverageMeter()

        for batch_idx, (x, y) in enumerate(trainloader):
            n_iter = round * self.args.VAE_local_epoch + epoch * len(trainloader) + batch_idx
            x, y = x.to(self.device), y.to(self.device)

            batch_size = x.size(0)
            re = self.args.VAE_re

            optimizer.zero_grad()
            out, _, gx, mu, logvar, _, _, _ = self.vae_model(x)

            cross_entropy = F.cross_entropy(out[: batch_size * 2], y.repeat(2))
            x_ce_loss = F.cross_entropy(out[batch_size * 2:], y)
            l1 = F.mse_loss(gx, x)
            l2 = cross_entropy
            l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l3 /= batch_size * 3 * self.args.VAE_z

            loss = 5 * re * l1 + self.args.VAE_ce * l2 + 0.5 * self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss

            loss.backward()
            optimizer.step()
            prec1, _, _, _, _ = accuracy(out[:batch_size].data, y[:batch_size].data, topk=(1, 5))

            loss_avg.update(loss.data.item(), batch_size)
            loss_rec.update(l1.data.item(), batch_size)
            loss_ce.update(cross_entropy.data.item(), batch_size)
            loss_kl.update(l3.data.item(), batch_size)
            top1.update(prec1.item(), batch_size)
    ## share_data_step附属函数
    def generate_data_by_vae(self):
        generate_dataloader = self.train_dataloader
        self.vae_model.to(self.device)
        self.vae_model.eval()

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(generate_dataloader):
    
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                _, _, gx, _, _, rx, rx_noise1, rx_noise2 = self.vae_model(x)

                if batch_idx == 0:
                    self.local_share_data1 = rx_noise1
                    self.local_share_data2 = rx_noise2
                    self.local_share_data_y = y
                else:
                    self.local_share_data1 = torch.cat((self.local_share_data1, rx_noise1))
                    self.local_share_data2 = torch.cat((self.local_share_data2, rx_noise2))
                    self.local_share_data_y = torch.cat((self.local_share_data_y, y))
    ## aggregate_sampled_client_vae附属函数
    def get_vae_para(self):
        return deepcopy(self.vae_model.cpu().state_dict())
    ## aggregate_sampled_client_vae附属函数
    def set_vae_para(self, para_dict):
        self.vae_model.load_state_dict(para_dict)

class Aggregator_VAE:
    def __init__(self, train_dataloader,test_dataloader,device, args, model_trainer, vae_model):

        self.trainer = model_trainer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.args= args
        self.vae_model = vae_model
    
    ##share_data_step附属函数
    def test_on_server_by_vae(self, round):
        self.vae_model.to(self.device)
        self.vae_model.eval()

        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                batch_size = x.size(0)

                out = self.vae_model.classifier_test(x)

                loss = F.cross_entropy(out, y)
                prec1, _ = accuracy(out.data, y)

                n_iter = (round - 1) * len(self.test_dataloader) + batch_idx
                test_acc_avg.update(prec1.data.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                total_acc_avg += test_acc_avg.avg
            total_acc_avg /= len(self.test_dataloader)
            return total_acc_avg
    ## aggregate_sampled_client_vae附属函数
    def set_vae_param(self, para_dict):
        self.vae_model.load_state_dict(para_dict)
    