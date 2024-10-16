import collections
import math
import sys
import time
import torch
import torch.utils
from .strategy import Strategy
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
from torchvision import transforms
import dataset
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate,print_log

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature, feature_dim=512):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.pre_lasts = torch.FloatTensor(self.n, feature_dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, pre_last, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.pre_lasts[self.ptr:self.ptr+b].copy_(pre_last.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.pre_lasts = self.pre_lasts.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, return_pre_last=False):
        pre_last = self.backbone(x)
        features = self.contrastive_head(pre_last)
        features = F.normalize(features, dim = 1)
        if return_pre_last:
            return features, pre_last
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out



""" Custom collate function """
def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))


def simclr_train(train_loader, model, criterion, optimizer, epoch):
        """ 
        Train according to the scheme from SimCLR
        https://arxiv.org/abs/2002.05709
        """
        # losses = AverageMeter('Loss', ':.4e')
        model.train()

        for i, batch in enumerate(train_loader):
            
            images = batch[0]
            images_augmented = batch[0]
            b, c, h, w = images.size()
            input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w) 
            input_ = input_.cuda(non_blocking=True)
            # targets = batch['target'].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            
            output = model(input_).view(b, 2, -1)
            loss = criterion(output)
            
            # losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        # images = batch['image'].cuda(non_blocking=True)
        # targets = batch['target'].cuda(non_blocking=True)
        images = batch[0].cuda(non_blocking=True)
        targets = batch[1].cuda(non_blocking=True)
        output, pre_last = model(images, return_pre_last=True)
        memory_bank.update(output, pre_last, targets)
        if i % 50 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


class TCM(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(TCM, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.n_pool = len(Y)
        self.NUM_QUERY = int(args.nQuery*self.n_pool/100) if args.nStart!= 100 else 0   
        self.MIN_CLUSTER_SIZE = 5
        self.MAX_NUM_CLUSTERS = 500
        self.K_NN = 20
        self.trans_flag = True
        self.uSet = np.arange(self.n_pool)[~self.idxs_lb]
        self.init_features_and_clusters()

    


    def simCLR(self):
        '''extract features with simCLR'''
        #get the model
        if(self.args.dataset in ['cifar10','cifar100']):
            from models.simclr_resnet_cifar import resnet18
            backbone = resnet18()
            model_kwargs = {'head': 'mlp' , 'features_dim':128 } 
            if(self.args.dataset == 'cifar10'):
                normal = transforms.Normalize(mean= [0.4914, 0.4822, 0.4465],std= [0.2023, 0.1994, 0.2010])
            else:
                normal = transforms.Normalize(mean= [0.5071, 0.4867, 0.4408],std= [0.2675, 0.2565, 0.2761])

            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale= [0.2,1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)
                ], p= 0.8),
                transforms.RandomGrayscale(p= 0.2),
                transforms.ToTensor(),
                normal
                ])

            test_transforms = transforms.Compose([
                transforms.CenterCrop(size=32),
                transforms.ToTensor(), 
                normal
                ])
            # for train_dataset, use augmentation
            train_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms)

            test_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,False,test_transforms)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=True, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(test_dataset)))
            base_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms) #Dataset w/o augs for knn eval
            base_dataloader = torch.utils.data.DataLoader(base_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            #memory bank, store the representation of train set and test set
            memory_bank_base = MemoryBank(len(base_dataset), 
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test = MemoryBank(len(test_dataset),
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test.cuda()
            memory_bank_base.cuda()

        elif(self.args.dataset == 'tinyimagenet'):
            from models.simclr_resnet_tinyimagenet import resnet18
            backbone = resnet18()
            model_kwargs = {'head': 'mlp' , 'features_dim':128 } 
            normal = transforms.Normalize(mean= [0.5071, 0.4867, 0.4408],std= [0.2675, 0.2565, 0.2761])
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size= 64, scale= [0.2,1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)
                ], p= 0.8),
                transforms.RandomGrayscale(p= 0.2),
                transforms.ToTensor(),
                normal
                ])

            test_transforms = transforms.Compose([
                transforms.CenterCrop(size= 64),
                transforms.ToTensor(), 
                normal
                ])
            
            train_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms)
            # train_dataset = dataset.AugmentedDataset(train_dataset)
            test_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,False,test_transforms)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=True, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(test_dataset)))
            base_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms) #Dataset w/o augs for knn eval
            base_dataloader = torch.utils.data.DataLoader(base_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            #memory bank, store the representation of train set and test set
            memory_bank_base = MemoryBank(len(base_dataset), 
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test = MemoryBank(len(test_dataset),
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test.cuda()
            memory_bank_base.cuda()

        elif (self.args.dataset == 'gtsrb'):
            # had resize to 32x32, use same model as cifar
            from models.simclr_resnet_cifar import resnet18
            backbone = resnet18()
            model_kwargs = {'head': 'mlp' , 'features_dim':128 } 
            normal = transforms.Normalize(mean= [0.5071, 0.4867, 0.4408],std= [0.2675, 0.2565, 0.2761])

            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale= [0.2,1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)
                ], p= 0.8),
                transforms.RandomGrayscale(p= 0.2),
                transforms.ToTensor(),
                normal
                ])

            test_transforms = transforms.Compose([
                transforms.CenterCrop(size=32),
                transforms.ToTensor(), 
                normal
                ])
            # for train_dataset, use augmentation
            train_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms)

            test_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,False,test_transforms)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=True, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(test_dataset)))
            base_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms) #Dataset w/o augs for knn eval
            base_dataloader = torch.utils.data.DataLoader(base_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            #memory bank, store the representation of train set and test set
            memory_bank_base = MemoryBank(len(base_dataset), 
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test = MemoryBank(len(test_dataset),
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test.cuda()
            memory_bank_base.cuda()
        elif (self.args.dataset == 'mnist'):
            import models.linear as ML
            backbone = ML.simclr_LeNet()
            model_kwargs = {'head': 'mlp' , 'features_dim':128 } 
            normal = transforms.Normalize(mean= [0.5071, ],std= [0.2675,])

            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale= [0.2,1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1)
                ], p= 0.8),
                transforms.RandomGrayscale(p= 0.2),
                transforms.ToTensor(),
                normal
                ])

            test_transforms = transforms.Compose([
                transforms.CenterCrop(size=32),
                transforms.ToTensor(), 
                normal
                ])
            # for train_dataset, use augmentation
            train_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms)

            test_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,False,test_transforms)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=True, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(test_dataset)))
            base_dataset = dataset.Dataset_uniform(self.args.dataset,self.args.data_path,True,train_transforms) #Dataset w/o augs for knn eval
            base_dataloader = torch.utils.data.DataLoader(base_dataset, num_workers= 8,
                                                            batch_size= 512, pin_memory=True, collate_fn=collate_custom,
                                                            drop_last=False, shuffle=False)
            #memory bank, store the representation of train set and test set
            memory_bank_base = MemoryBank(len(base_dataset), 
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test = MemoryBank(len(test_dataset),
                                128,
                                self.args.n_class , 0.1)
            memory_bank_test.cuda()
            memory_bank_base.cuda()
        else:
            raise NotImplementedError
        model = ContrastiveModel(backbone, **model_kwargs)
        model = model.cuda()
        # CUDNN
        torch.backends.cudnn.benchmark = True
        # criterion
        criterion = SimCLRLoss(0.1)
        criterion = criterion.cuda()
        # optimizer
        params = model.parameters()
        optimizer = torch.optim.SGD(params, nesterov= False, weight_decay= 0.0001, momentum =0.9, lr= 0.4)
        # training
        tm = time.time()*1000
        for epoch in range(0, 5):
            print('>>>>> epoch = {}'.format(epoch))
            lr = 0.4 
            eta_min = lr * (0.1 ** 3)
            lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / 500)) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr  
            # train             
            simclr_train(train_dataloader, model, criterion, optimizer, epoch)
            
            fill_memory_bank(base_dataloader, model, memory_bank_base)
            
        # Mine the topk nearest neighbors at the very end (Train) 
        # These will be served as input to the SCAN loss.
        fill_memory_bank(base_dataloader, model, memory_bank_base)
        topk = 20
        train_topk_indices, acc = memory_bank_base.mine_nearest_neighbors(topk)    
        
        train_features = memory_bank_base.pre_lasts.cpu().numpy()
        test_features = memory_bank_test.pre_lasts.cpu().numpy()
        np.save(self.args.dataset + 'pretext_features.npy', train_features)
        np.save(self.args.dataset + 'pretext_test_features.npy', test_features)
        tm = time.time()*1000-tm
        print_log("\nsimclr use time = {:02f}]\n".format(tm), self.args.log)

        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        fill_memory_bank(test_dataloader, model, memory_bank_test)
        topk = 5
        test_topk_indices, acc = memory_bank_test.mine_nearest_neighbors(topk)

        return train_features, test_features, train_topk_indices, test_topk_indices

    


    def init_features_and_clusters(self):
        num_clusters = min(len(self.idxs_lb) + self.NUM_QUERY, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering.')
        self.features = np.load('pretext_features.npy')
        #self.features, _, _, _ = self.simCLR()  # train set features
        self.clusters = kmeans(self.features, num_clusters= num_clusters)
        print(f'Finished clustering into {num_clusters} clusters.')

    
    def query_typiclust(self,n):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.concatenate([self.idxs_lb, self.uSet]).astype(int)
        features = self.features[relevant_indices]
        labels = np.copy(self.clusters[relevant_indices])
        existing_indices = np.arange(len(self.idxs_lb))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        

        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        for i in range(n):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        print(f'Finished the selection of {len(activeSet)} samples.')
        self.trans_flag = False
        return activeSet
    
    def query_margin(self,n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]
    
    def query(self,n):
        if self.trans_flag:
            print('>>>> query by tpiclust >>>')
            return self.query_typiclust(n)
        else:
            print('>>>> query by margin >>>')
            return self.query_margin(n)

        


