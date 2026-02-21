import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from utils.inc_net import IncrementalNet_CausalETF as IncrementalNet
from methods.base_cvpr import BaseLearner
from utils.dmc import partition_data, DatasetSplit, average_weights, average_bias_vector, setup_seed, pil_loader
import copy, wandb
import os, math
import pickle
from PIL import Image
import random
from itertools import chain, combinations
import shutil
from torchvision import transforms
from collections import defaultdict
from torchvision.transforms import ColorJitter
from PIL import ImageOps
from PIL import ImageDraw
import json
import ipdb
import copy
import logging


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices, transform):

        self.images = dataset.images[indices]
        self.labels = dataset.labels[indices]
        self.transform = transform

    def __getitem__(self, idx):

        image = self.transform(Image.fromarray(self.images[idx]))  # 通过索引映射到原始数据集的索引
        label = self.labels[idx]
        return idx, image, label

    def __len__(self):

        return len(self.indices)


def insert_cifar_on_background(foreground, background, tttt=200, scale=[0.8, 0.9]):

    background = apply_color_jitter(foreground)  # 风格一致
    
    bg_w, bg_h = background.size
    fg_w, fg_h = foreground.size

    new_w, new_h = int(fg_w * scale[0]), int(fg_h * scale[1])
    foreground = foreground.resize((new_w, new_h), Image.ANTIALIAS)

    background = background.convert("RGBA")
    foreground = foreground.convert("RGBA")

    alpha_mask = Image.new("L", (new_w, new_h), tttt)
    foreground.putalpha(alpha_mask)

    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    composed = background.copy()
    composed.paste(foreground, (x, y), foreground)  

    return composed.convert("RGB")  


def apply_color_jitter(image):

    jitter = ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4)
    return jitter(image)

class CustomConcatDataset(ConcatDataset):

    def __init__(self, datasets, transform, args, logger):
        if args['dataset'] == 'tiny_imagenet':
            self.use_path = True
        else:
            self.use_path = False
        datasets = [*datasets[0], *[item for _ in range(args['n']) for item in datasets[1]]]

        self.labels = np.concatenate([ds.labels for ds in datasets])
        self.images = np.concatenate([ds.images for ds in datasets])

        self.local_len = len(datasets[0].labels)
        self.transform = transform
        self.scale_list = [0.8, 0.85, 0.9, 0.95]
        self.args = args

        LD = label_distribution(self.labels)
        logger.info("Label Distribution: %s" % str(LD))
        print(LD)

    def __getitem__(self, idx):

        if self.use_path:
            image = pil_loader(self.images[idx])
        else:
            image = Image.fromarray(self.images[idx])

        label = self.labels[idx]
        if idx >= self.local_len:
            is_com = random.choice(['NO', 'YES'])
            if is_com == 'YES':
                background_ids = random.sample(range(len(self.labels)), k=5)
                scale_0 = random.choice(self.scale_list)
                scale_1 = random.choice(self.scale_list)
                image = insert_cifar_on_background(image, None, tttt=self.args['tttt'], scale=[scale_0, scale_1])

        img = self.transform(image)
        return idx, img, label

    def __len__(self):
        return len(self.labels)


def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [-m / s for m, s in zip(mean, std)]
        _std = [1 / s for s in std]
    else:
        _mean = mean
        _std = std

    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)


def label_distribution(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


class FLDataSelector:
    def __init__(self, num_clients, local_features, query_budget):

        self.num_clients = num_clients
        self.local_features = local_features
        self.query_budget = query_budget
        self.P = [self.generate_random_orthogonal_matrix(F.shape[0]) for F in local_features]  # 生成每个客户端的 P_i
        self.Q = self.generate_random_orthogonal_matrix(local_features[0].shape[1])  # 生成全局的 Q

    def generate_random_orthogonal_matrix(self, size):

        random_matrix = torch.randn(size, size)
        Q, _ = torch.linalg.qr(random_matrix)  # QR 分解生成正交矩阵
        return Q

    def mask_local_data(self):

        masked_features = []
        for i, F in enumerate(self.local_features):
            masked_F = torch.matmul(torch.matmul(self.P[i], F), self.Q)  # X'_i = P_i X_i Q
            masked_features.append(masked_F)
        return masked_features

    def compute_local_leverage_scores(self, masked_features):

        leverage_scores = []
        for F in masked_features:
            U, _, _ = torch.svd(F)
            tau_i = torch.norm(U, dim=1) ** 2
            tau_i /= tau_i.sum()
            leverage_scores.append(tau_i)
        return leverage_scores

    def aggregate_leverage_scores(self, leverage_scores):

        client_weights = torch.tensor([1.0 / self.num_clients] * self.num_clients)  # 每个客户端的权重均衡

        all_scores = [tau * client_weights[i] for i, tau in enumerate(leverage_scores)]

        all_scores = torch.cat(all_scores)
        p = all_scores / all_scores.sum()
        return p

    def sample_data(self, p):

        min_samples_per_client = self.query_budget // self.num_clients
        client_selected_indices = {i: [] for i in range(self.num_clients)}

        start_idx = 0
        for i in range(self.num_clients):
            end_idx = start_idx + len(self.local_features[i])
            client_p = p[start_idx:end_idx]
            sampled = torch.multinomial(client_p, min_samples_per_client, replacement=True)
            client_selected_indices[i] = sampled.cpu().numpy()
            start_idx = end_idx

        remaining_budget = self.query_budget - sum(len(v) for v in client_selected_indices.values())
        if remaining_budget > 0:
            extra_samples = torch.multinomial(p, remaining_budget, replacement=True).tolist()
            for idx in extra_samples:
                for i in range(self.num_clients):
                    start_idx = sum(len(self.local_features[j]) for j in range(i))
                    end_idx = start_idx + len(self.local_features[i])
                    if start_idx <= idx < end_idx:
                        client_selected_indices[i].append(idx)
                        break

        return client_selected_indices


class FEAT(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.class_order = torch.tensor(args["class_order"], device=args["gpu"])
        self.args = args
        self.r = args['r']
        self.ltc = args['ltc']
        self.transform, self.normalizer = self._get_norm_and_transform(self.args["dataset"])
        self.selected_data_indices = []
        self.retained_ds_all = [[] for _ in range(args['num_users'])]

        self.PH, self.PT, self.bar_eH_G, self.r_H, self.r_T = None, None, None, None, None

    def _get_norm_and_transform(self, dataset):

        if dataset == "cifar100":
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
            train_transform = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        elif dataset == "cifar10":
            data_normalize = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            train_transform = transforms.Compose([
                # transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=63 / 255),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])

        elif dataset == "tiny_imagenet":
            data_normalize = dict(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            train_transform = transforms.Compose([
                # transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**dict(data_normalize)),
            ])
        return train_transform, Normalizer(**dict(data_normalize))

    def _get_client_dataset(self, client_idx):

        self.train_dataset, _ = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        client_indices = self.user_groups[client_idx]

        client_dataset = DatasetSplit(self.train_dataset, client_indices)
        return client_dataset

    def after_task(self):

        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        test_acc = self._compute_accuracy(self._old_network, self.test_loader, self.PH, self.PT, self.bar_eH_G, self.r_H, self.r_T)
        self.logger.info("After Task: %d,  Test ACC: %s" % (self._cur_task, str(test_acc)))

        print("After Test Acc: %s" % test_acc)

    def _select_data_for_retention(self):

        all_client_features = []
        all_client_indices = []
        for client_idx in range(self.args["num_users"]):
            try:
                client_features, client_indices = self._extract_client_features(client_idx)
                if len(client_features) > 0:
                    all_client_features.append(client_features)
                    all_client_indices.append(client_indices)
            except ValueError as e:
                print(f"Error extracting features for client {client_idx}: {e}")
                continue

        if len(all_client_features) == 0:
            print("No features extracted from any client. Skipping data selection.")
            return

        selector = FLDataSelector(self.args["num_users"], all_client_features,
                                  (self._total_classes - self._known_classes) * self.args["mem_size"])
        masked_features = selector.mask_local_data()

        leverage_scores = selector.compute_local_leverage_scores(masked_features)

        p = selector.aggregate_leverage_scores(leverage_scores)

        selected_indices = selector.sample_data(p)

        self.selected_data_indices = selected_indices

    def _extract_client_features(self, client_idx):

        if not hasattr(self, "_known_classes") or not hasattr(self, "_total_classes"):
            raise ValueError("_known_classes or _total_classes is not defined.")

        client_dataset = self._get_client_dataset(client_idx)
        local_train_loader = DataLoader(
            client_dataset,
            batch_size=self.args["local_bs"],
            shuffle=False,
            num_workers=self.args["num_worker"],
            pin_memory=True,
            multiprocessing_context=self.args["mulc"],
            persistent_workers=True,
            drop_last=False
        )
        features = []
        indices = []
        self._network.eval()
        with torch.no_grad():
            for batch_idx, (_, images, labels) in enumerate(local_train_loader):
                images = images.cuda()
                output_list = self._network(images)
                feature = output_list["att"]
                features.append(feature.cpu())
                start_idx = batch_idx * self.args["local_bs"]
                end_idx = start_idx + images.size(0)
                indices.extend(range(start_idx, end_idx))

        features = torch.cat(features, dim=0)
        return features, indices

    def _get_retained_dataset(self, client_idx):

        client_dataset = self._get_client_dataset(client_idx)
        retained_dataset = IndexedDataset(client_dataset, self.selected_data_indices[client_idx], self.transform)

        return retained_dataset

    def incremental_train(self, data_manager, logger):
        self.logger = logger

        setup_seed(self.seed)
        self.data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes, num_head=self.args["num_head"], tau=self.args["tau"], alpha=self.args["alpha"], gamma=self.args["gamma"])
        self._network.update_fc(self._total_classes)
        self._network.cuda()
        print("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 获取当前任务的数据
        train_dataset, train_data_indices = self.data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )

        test_dataset, _ = self.data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"],
                                      multiprocessing_context=self.args["mulc"], persistent_workers=True)

        if self._cur_task > 0:
            old_test_dataset, _ = self.data_manager.get_dataset(
                np.arange(0, self._known_classes), source="test", mode="test"
            )
            self.old_loader = DataLoader(
                old_test_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"], persistent_workers=True
            )
            new_dataset, _ = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="test", mode="test"
            )
            self.new_loader = DataLoader(
                new_dataset, batch_size=256, shuffle=False, num_workers=self.args["num_worker"],
                multiprocessing_context=self.args["mulc"], persistent_workers=True
            )

        self._fl_train(train_dataset, train_data_indices, self.test_loader)

    def _fl_train(self, train_dataset, train_data_indices, test_loader):

        self._network.cuda()
        #         ipdb.set_trace()
        self.best_model = None
        self.lowest_loss = np.inf

        prog_bar = tqdm(range(self.args["com_round"]))
        optimizer = torch.optim.SGD(self._network.parameters(), lr=self.args['local_lr'], momentum=0.9,
                                    weight_decay=self.args['weight_decay'])
        if self.args["dataset"] == "tiny_imagenet" or self.args["dataset"] == "imagenet":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["com_round"], eta_min=1e-3)

        user_groups, _ = partition_data(train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"])
        self.user_groups = user_groups

        for _, com in enumerate(prog_bar):
            local_weights = []
            # m = max(int(self.args["frac"] * self.args["num_users"]), 1)
            # idxs_users = np.random.choice(range(self.args["num_users"]), m, replace=False)
            idxs_users = range(self.args["num_users"])
            loss_weight = []
            priors_list = []
            for idx in idxs_users:
                local_train_ds_i = DatasetSplit(train_dataset, self.user_groups[idx])
                if self._cur_task > 0:

                    client_bboxes = None
                    local_train_ds_i = CustomConcatDataset([[local_train_ds_i], self.retained_ds_all[idx]], self.transform, self.args, self.logger)
                    print('####', local_train_ds_i.labels.shape)
                local_train_loader = DataLoader(local_train_ds_i, batch_size=self.args["local_bs"], shuffle=True,
                                                num_workers=self.args["num_worker"], pin_memory=True,
                                                multiprocessing_context=self.args["mulc"], persistent_workers=True)

                if self._cur_task == 0:
                    w, total_loss = self._local_update(copy.deepcopy(self._network), local_train_loader,
                                                       scheduler.get_last_lr()[0])
                else:
                    w, total_loss, priors, P_H, P_T = self._local_finetune(self._old_network, copy.deepcopy(self._network),
                                                                     local_train_loader, self._cur_task, idx, scheduler.get_last_lr()[0])
                    priors_list.append(priors)

                local_weights.append(copy.deepcopy(w))
                loss_weight.append(total_loss)

                del local_train_loader, w
                torch.cuda.empty_cache()
            scheduler.step()
            sum_loss = sum(loss_weight)
            if sum_loss < self.lowest_loss:
                self.lowest_loss = sum_loss
                self.best_model = copy.deepcopy(self._network.state_dict())

            global_weights = average_weights(local_weights)
            self._network.load_state_dict(global_weights)

            if com % 1 == 0 and com < self.args["com_round"]:
                test_acc = self._compute_accuracy(self._network, test_loader)
                if self._cur_task > 0:
                    test_old_acc = self._compute_accuracy(copy.deepcopy(self._network), self.old_loader)
                    test_new_acc = self._compute_accuracy(copy.deepcopy(self._network), self.new_loader)
                    print("Task {}, Test_accy {:.2f} O {} N {}".format(self._cur_task, test_acc, test_old_acc,
                                                                       test_new_acc))
                info = (
                    "Task {}, Epoch {}/{} =>  Test_accy {:.2f}".format(self._cur_task, com + 1, self.args["com_round"],
                                                                       test_acc))
                self.logger.info(info)
                prog_bar.set_description(info)
                
        if self._cur_task > 0:
            global_priors = aggregate_ecc_priors_from_clients(priors_list, eps=1e-8)
            self.bar_eH_G = global_priors["bar_eH_G"]
            self.PH = P_H
            self.PT = P_T
            self.r_H = self._total_classes - self._known_classes - 1
            self.r_T = self._known_classes - 1
            
        self._network.load_state_dict(self.best_model)
        if self._cur_task < self.tasks - 1:
            self._select_data_for_retention()
            for idx in idxs_users:
                #
                local_retained_ds = self._get_retained_dataset(idx)
                self.retained_ds_all[idx].append(local_retained_ds)
            # #
                client_retained_indices = self.selected_data_indices[idx]
                client_indices_all = train_data_indices[self.user_groups[idx]]

        del self.best_model
        torch.cuda.empty_cache()

    def _local_update(self, model, train_data_loader, lr):

        model.train()
        total_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
        prototypes = model.fc.weight
        # ipdb.set_trace()
        for iter in range(self.args["local_ep"]):
            epoch_lossce_collector = []
            epoch_lossrd_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                output_list = model(images)
                output = output_list["logits"]
                loss_ce = F.cross_entropy(output, labels)
                feats = output_list["features"]
                loss_rd = gsa_distill_loss(feats, labels, prototypes)
                loss = loss_ce + self.args["kd"] * loss_rd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter == 0:
                    total_loss += loss.detach()
                epoch_lossce_collector.append(loss_ce.item())
                epoch_lossrd_collector.append(loss_rd.item())
            epoch_lossce = sum(epoch_lossce_collector) / len(epoch_lossce_collector)
            epoch_lossrd = sum(epoch_lossrd_collector) / len(epoch_lossrd_collector)
            self.logger.info('Epoch: %d Loss CE: %f  Loss RD: %f ' % (iter, epoch_lossce, epoch_lossrd))
        return model.state_dict(), total_loss

    def _local_finetune(self, teacher, model, train_data_loader, task_id, client_id, lr):

        model.train()
        teacher.eval()
        total_loss = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
        prototypes = model.fc.weight
        tail_class_ids = list(range(self._known_classes))
        for it in range(self.args["local_ep"]):
            epoch_lossce_collector = []
            epoch_lossrd_collector = []
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                # print('labels:', labels)
                output_list = model(images)
                output = output_list["logits"]
                loss_ce = F.cross_entropy(output, labels)
                feats = output_list["features"]
                loss_rd = gsa_distill_loss(feats, labels, prototypes)
                loss = loss_ce + self.args["kd"] * loss_rd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if it == 0:
                    total_loss += loss.detach()
                epoch_lossce_collector.append(loss_ce.item())
                epoch_lossrd_collector.append(loss_rd.item())
            epoch_lossce = sum(epoch_lossce_collector) / len(epoch_lossce_collector)
            epoch_lossrd = sum(epoch_lossrd_collector) / len(epoch_lossrd_collector)
            self.logger.info('Epoch: %d Loss CE: %f  Loss RD: %f ' % (it, epoch_lossce, epoch_lossrd))
        print('prototypes.shape', prototypes.shape)
        print('self._known_classes', self._known_classes)
        tail_class_ids = list(range(self._known_classes))
        head_class_ids = list(range(self._known_classes, self._total_classes))
        priors, P_H, P_T = estimate_ecc_priors(model=model, data_loader=train_data_loader, head_class_ids=head_class_ids, tail_class_ids=tail_class_ids, \
                                                        ema_decay=float(self.args["ema_decay"]), feature_key="features")
        return model.state_dict(), total_loss, priors, P_H, P_T


def gsa_distill_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    tau: float = 0.07,
    eps: float = 1e-8,
    include_self: bool = True,
) -> torch.Tensor:

    device = features.device
    B, d = features.shape

    f = F.normalize(features, dim=1)        # [B, d]
    W = F.normalize(prototypes, dim=0)      # [d, C_t]

    MF = torch.matmul(f, f.t())             # [B, B]

    Py = W[:, labels].t().contiguous()      # [B, d]
    Py = F.normalize(Py, dim=1)
    MP = torch.matmul(Py, Py.t())           # [B, B]

    if not include_self:
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        MF = MF.masked_fill(diag_mask, -float('inf'))
        MP = MP.masked_fill(diag_mask, -float('inf'))

    logPF = F.log_softmax(MF / max(tau, eps), dim=1)        # [B, B]
    PF    = logPF.exp()
    logPP = F.log_softmax(MP / max(tau, eps), dim=1).detach()

    row_kl = (PF * (logPF - logPP)).sum(dim=1)              # [B]


    uniq, inv = torch.unique(labels, return_inverse=True)  # uniq: [C_B], inv: [B] in [0..C_B-1]
    Cb = uniq.numel()
    sum_per_cls = torch.zeros(Cb, device=device).scatter_add_(0, inv, row_kl)
    cnt_per_cls = torch.zeros(Cb, device=device).scatter_add_(0, inv, torch.ones_like(row_kl))
    loss = (sum_per_cls / (cnt_per_cls + eps)).mean()

    return loss




def _build_projector_from_W(W_part: torch.Tensor) -> torch.Tensor:
    """
    W_part: [d, C_part], column is a (normalized) prototype
    return: P = W (W^T W)^{+} W^T  \in R^{dxd}
    """

    G = W_part.t() @ W_part                  # [C_part, C_part]
    P = W_part @ torch.linalg.pinv(G) @ W_part.t()   # [d, d]

    return 0.5 * (P + P.t())


@torch.no_grad()
def estimate_ecc_priors(
    model,
    data_loader,
    head_class_ids,
    tail_class_ids,
    ema_decay: float = 0.9,
    feature_key: str = "features",
):

    device = next(model.parameters()).device

    W = F.normalize(model.fc.weight.detach(), dim=1).t()  # [d, C_t]
    W_H = W[:, head_class_ids] if len(head_class_ids) > 0 else None
    W_T = W[:, tail_class_ids] if len(tail_class_ids) > 0 else None
    if W_H is None or W_T is None:
        return {"bar_eH_T": 0.0, "bar_eT_T": 0.0, "nT": 0}, None, None

    P_H = _build_projector_from_W(W_H)   # [d, d]
    P_T = _build_projector_from_W(W_T)   # [d, d]

    r_H = max(len(head_class_ids) - 1, 1)   # simplex rank
    r_T = max(len(tail_class_ids) - 1, 1)

    bar_eH = torch.tensor(0.0, device=device)
    bar_eT = torch.tensor(0.0, device=device)
    nT = 0
    tail_set = set(int(x) for x in tail_class_ids)

    model.eval()
    for _, images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mask = torch.tensor([int(y.item()) in tail_set for y in labels],
                            device=device, dtype=torch.bool)
        if mask.sum() == 0:
            continue

        out = model(images)
        feat = out.get(feature_key, None)
        if feat is None:
            raise RuntimeError(f"model(images) '{feature_key}'。")
        f = F.normalize(feat[mask], dim=1)  # [B_t, d]

        # rank-normalized energies
        eH = (f @ (P_H @ f.t())).diagonal() / r_H
        eT = (f @ (P_T @ f.t())).diagonal() / r_T

        mH = eH.mean()
        mT = eT.mean()
        bar_eH = (1 - ema_decay) * bar_eH + ema_decay * mH
        bar_eT = (1 - ema_decay) * bar_eT + ema_decay * mT
        nT += int(mask.sum().item())

    return {
        "bar_eH_T": float(bar_eH.item()),
        "bar_eT_T": float(bar_eT.item()),
        "nT": nT,
    }, P_H, P_T


def aggregate_ecc_priors_from_clients(priors_list, eps=1e-12):

    num = 0
    sum_eH = 0.0
    sum_eT = 0.0
    for p in priors_list:
        nT = int(p.get("nT", 0))
        if nT <= 0:
            continue
        sum_eH += float(p["bar_eH_T"]) * nT
        sum_eT += float(p["bar_eT_T"]) * nT
        num    += nT

    if num == 0:
        return None

    bar_eH_G = sum_eH / (num + eps)
    bar_eT_G = sum_eT / (num + eps)
    return {"bar_eH_G": bar_eH_G, "bar_eT_G": bar_eT_G, "nT_sum": num}











