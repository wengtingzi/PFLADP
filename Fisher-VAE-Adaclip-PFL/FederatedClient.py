import torch
import torch.optim as optim
from utils import compute_fisher_diag
import torch.nn.functional as F

class FederatedClient:
    def __init__(self, model, train_loader, test_loader, device,global_model,args):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.global_model = global_model
        self.device = device
        self.args = args

    def local_update(self, vae_train_mixed_dataloader=None):
        fisher_threshold = self.args.fisher_threshold
        self.model = self.model.to(self.device)
        self.global_model = self.global_model.to(self.device)

        w_glob = [param.clone().detach() for param in self.global_model.parameters()]

        fisher_diag = compute_fisher_diag(self.model, self.train_loader)

        u_loc, v_loc = [], []
        for param, fisher_value in zip(self.model.parameters(), fisher_diag):
            u_param = (param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (param * (fisher_value <= fisher_threshold)).clone().detach()
            u_loc.append(u_param)
            v_loc.append(v_param)

        u_glob, v_glob = [], []
        for global_param, fisher_value in zip(self.global_model.parameters(), fisher_diag):
            u_param = (global_param * (fisher_value > fisher_threshold)).clone().detach()
            v_param = (global_param * (fisher_value <= fisher_threshold)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)

        for u_param, v_param, model_param in zip(u_loc, v_glob, self.model.parameters()):
            model_param.data = u_param + v_param

        saved_u_loc = [u.clone() for u in u_loc]

        def custom_loss(outputs, labels, param_diffs, reg_type):
            ce_loss = F.cross_entropy(outputs, labels)
            if reg_type == "R1":
                reg_loss = (self.args.lambda_1 / 2) * torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))

            elif reg_type == "R2":
                C = self.args.clipping_bound
                norm_diff = torch.sum(torch.stack([torch.norm(diff) for diff in param_diffs]))
                reg_loss = (self.args.lambda_2 / 2) * torch.norm(norm_diff - C)

            else:
                raise ValueError("Invalid regularization type")

            return ce_loss + reg_loss

        optimizer1 = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # Train with original data
        for epoch in range(self.args.local_epoch):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer1.zero_grad()
                outputs = self.model(data)
                param_diffs = [u_new - u_old for u_new, u_old in zip(self.model.parameters(), w_glob)]
                loss = custom_loss(outputs, labels, param_diffs, "R1")
                loss.backward()
                with torch.no_grad():
                    for model_param, u_param in zip(self.model.parameters(), u_loc):
                        model_param.grad *= (u_param != 0)
                optimizer1.step()

        optimizer2 = optim.Adam(self.model.parameters(), lr=self.args.lr)
        if vae_train_mixed_dataloader is not None:
            vae_dataloader = vae_train_mixed_dataloader
        else:
            vae_dataloader = self.train_loader

        # Train with VAE generated data
        for epoch in range(self.args.local_epoch):
            for batch_idx, (x1, x2, x3, y1, y2, y3) in enumerate(vae_dataloader):
                x1, x2, x3 = x1.to(self.device), x2.to(self.device), x3.to(self.device)
                y1, y2, y3 = y1.to(self.device), y2.to(self.device), y3.to(self.device)
                batch_size = x1.size(0)
                data = torch.cat((x1, x2, x3), dim=0)
                labels = torch.cat((y1, y2, y3), dim=0)
                optimizer2.zero_grad()
                outputs = self.model(data)
                param_diffs = [model_param - w_old for model_param, w_old in zip(self.model.parameters(), w_glob)]
                loss = custom_loss(outputs, labels, param_diffs, "R2")
                loss.backward()
                with torch.no_grad():
                    for model_param, v_param in zip(self.model.parameters(), v_glob):
                        model_param.grad *= (v_param != 0)
                optimizer2.step()

        with torch.no_grad():
            update = [(new_param - old_param).clone() for new_param, old_param in zip(self.model.parameters(), w_glob)]

        self.model = self.model.to('cpu')
        return update

    def test(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        num_data = 0
        correct = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                num_data += labels.size(0)

        accuracy = 100.0 * correct / num_data
        self.model = self.model.to('cpu')

        return accuracy