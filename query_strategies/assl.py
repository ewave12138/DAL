from math import sqrt
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
import time
from .strategy import Strategy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from copy import deepcopy

pseudo_label_threshold = 0.95
pseudo_label_temperature = 1



def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


class RandAugmentMC(object):
    def AutoContrast(self, img, **kwarg):
        return PIL.ImageOps.autocontrast(img)

    def Brightness(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Brightness(img).enhance(v)

    def Color(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Color(img).enhance(v)

    def Contrast(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Contrast(img).enhance(v)

    def Cutout(self, img, v, max_v, bias=0):
        if v == 0:
            return img
        v = float(v) * max_v / 10 + bias
        v = int(v * min(img.size))
        return self.CutoutAbs(img, v)

    def CutoutAbs(self, img, v, **kwarg):
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        xy = (x0, y0, x1, y1)
        # gray
        color = (127) if self.channels==1 else (127, 127, 127)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

    def Equalize(self, img, **kwarg):
        return PIL.ImageOps.equalize(img)

    def Identity(self, img, **kwarg):
        return img

    def Invert(self, img, **kwarg):
        return PIL.ImageOps.invert(img)

    def Posterize(self, img, v, max_v, bias=0):
        v = int(v * max_v / 10) + bias
        return PIL.ImageOps.posterize(img, v)

    def Rotate(self, img, v, max_v, bias=0):
        v = int(v * max_v / 10) + bias
        if random.random() < 0.5:
            v = -v
        return img.rotate(v)

    def Sharpness(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Sharpness(img).enhance(v)

    def ShearX(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

    def ShearY(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

    def Solarize(self, img, v, max_v, bias=0):
        v = int(v * max_v / 10) + bias
        return PIL.ImageOps.solarize(img, 256 - v)

    def SolarizeAdd(self, img, v, max_v, bias=0, threshold=128):
        v = int(v * max_v / 10) + bias
        if random.random() < 0.5:
            v = -v
        img_np = np.array(img).astype(np.int)
        img_np = img_np + v
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold)

    def TranslateX(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        if random.random() < 0.5:
            v = -v
        v = int(v * img.size[0])
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    def TranslateY(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        if random.random() < 0.5:
            v = -v
        v = int(v * img.size[1])
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

    def __init__(self, n, m, size,channels):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.size = size
        self.channels = channels
        self.augment_pool = [
            (self.AutoContrast, None, None),
            (self.Brightness, 0.9, 0.05),
            (self.Color, 0.9, 0.05),
            (self.Contrast, 0.9, 0.05),
            (self.Equalize, None, None),
            (self.Identity, None, None),
            (self.Posterize, 4, 4),
            (self.Rotate, 30, 0),
            (self.Sharpness, 0.9, 0.05),
            (self.ShearX, 0.3, 0),
            (self.ShearY, 0.3, 0),
            (self.Solarize, 256, 0),
            (self.TranslateX, 0.3, 0),
            (self.TranslateY, 0.3, 0)
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = self.CutoutAbs(img, int(self.size*0.5))
        return img


class TransformUDA(object):
    def __init__(self, mean, std,size,channels):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10,size=size,channels=channels)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def save_to_txt(file_path, content):
    """
    file_path (str)
    content (str)
    """
    try:
        with open(file_path, 'a') as file:  
            file.write(content + '\n')  
        print(f"content saved to {file_path} successfully")
    except Exception as e:
        print(f"content save failure: {e}")


class assl(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """

    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(assl, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.n_pool  = len(Y)



    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        query_score = self.ucb_inconsist * self.ucb_uncertain

        # print('>>>>> len of query score ={}'.format(len(query_score)))

        indices_partition = np.argpartition(query_score, -n)[-n:]
        indices_partition = indices_partition[np.argsort(query_score[indices_partition])][::-1]

        # print('n = {}'.format(n))
        # print('len of indices_partition = {}'.format(len(indices_partition)))

        # map into the whole  train dataset
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
      
        ori_map = [idxs_unlabeled[i] for i in indices_partition]
        return ori_map      #  return  a list 

    
    def calculate_pseudo_el2n(self, logits, pseudo_labels):
        el2n_score = torch.norm(logits - F.one_hot(pseudo_labels, num_classes=logits.size(-1)).float(), p=2, dim=-1)
        return el2n_score

    def compute_data_inconsistency(self, weak_outputs, strong_outputs):
        """
        Data-inconsistency
        """
        weak_probs = F.softmax(weak_outputs, dim=0)  
        strong_probs = F.softmax(strong_outputs, dim=0)
        
        kl_div = F.kl_div(F.log_softmax(weak_outputs, dim=0), strong_probs, reduction='batchmean')
        
        return kl_div


    def update_ucb(self, loader_ulb, ema_decay, cu, ci):
        for batch_idx, (images, y, index) in enumerate(loader_ulb):
            x_weak, x_strong = images
            nan_mask = torch.isnan(x_weak)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())
            logits_weak, _ = self.clf(x_weak)
            logits_strong, _ = self.clf(x_strong)
            logits_weak = torch.tensor(logits_weak)
            logits_strong = torch.tensor(logits_strong)
            pseudo_label_w = torch.softmax(logits_weak.detach(), dim=-1)
            pseudo_label_s = torch.softmax(logits_strong.detach(), dim=-1)

            max_probs, targets_u_weak = torch.max(pseudo_label_w, dim=-1)
            x_el2n = self.calculate_pseudo_el2n(logits_weak,targets_u_weak).cpu().numpy()

            # self.ut[index]= ema_decay* x_el2n + (1-ema_decay)*self.ut[index]
            # self.vt[index]= ema_decay*((x_el2n-self.ut[index])**2) +(1-ema_decay)* self.vt[index]
            # self.ucb_uncertain[index]= self.ut[index]+ cu* sqrt(self.vt[index])
            # x_kl = (self.compute_data_inconsistency(pseudo_label_w,pseudo_label_s)+ self.compute_data_inconsistency(pseudo_label_s,pseudo_label_w))/2.0
            # self.it[index] = ema_decay*x_kl +(1- ema_decay)* self.it[index]
            # self.vit[index] = ema_decay*((x_kl- self.it[index])**2)+ (1-ema_decay)*self.vit[index]
            # self.ucb_inconsist[index]= self.it[index]+ ci* sqrt(self.vit[index])

        
        for i, idx in enumerate(index):
            self.ut[idx] = ema_decay * x_el2n[i] + (1 - ema_decay) * self.ut[idx]
            self.vt[idx] = ema_decay * ((x_el2n[i] - self.ut[idx]) ** 2) + (1 - ema_decay) * self.vt[idx]
            self.ucb_uncertain[idx] = self.ut[idx] + cu * sqrt(self.vt[idx])
            
            x_kl = (self.compute_data_inconsistency(logits_weak[i], logits_strong[i]) +
                    self.compute_data_inconsistency(logits_strong[i], logits_weak[i])) / 2.0
            self.it[idx] = ema_decay * x_kl + (1 - ema_decay) * self.it[idx]
            self.vit[idx] = ema_decay * ((x_kl - self.it[idx]) ** 2) + (1 - ema_decay) * self.vit[idx]
            self.ucb_inconsist[idx] = self.it[idx] + ci * sqrt(self.vit[idx])

           
            

    def _train(self, epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer):
        self.clf.train()
        accFinal = 0.
        train_loss = 0.
        total_steps = 35000
        iter_unlabeled = iter(loader_tr_unlabeled)
        for batch_idx, (x, y, idxs) in enumerate(loader_tr_labeled):
            try:
                (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
            except StopIteration:
                iter_unlabeled = iter(loader_tr_unlabeled)
                (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
            input_all = torch.cat((x, inputs_u, inputs_u2)).to(self.device)
            y = y.to(self.device)
            output_all, _ = self.clf(input_all)
            output_sup = output_all[:len(x)]
            output_unsup = output_all[len(x):]
            output_u, output_u2 = torch.chunk(output_unsup, 2)
            loss = F.cross_entropy(output_sup, y, reduction='mean')    # loss for supervised learning
            pseudo_label = torch.softmax(output_u.detach() / pseudo_label_temperature, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(pseudo_label_threshold).float()
            masked_loss = F.cross_entropy(output_u2, max_idx, reduction="none") * mask
            unsup_loss = masked_loss.mean()

            loss += unsup_loss
            train_loss += loss.item()
            accFinal += torch.sum((torch.max(output_sup, 1)[1] == y).float()).data.item()

            # exit()
            optimizer.zero_grad()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()


            if batch_idx % 10 == 0:
                print("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))
        self.update_ucb(loader_tr_unlabeled, 0.8, 0.5, 2.0)
        return accFinal / len(loader_tr_labeled.dataset.X), train_loss

    def train(self, alpha=0.1, n_epoch=10, batch_ratio=0.6):
        self.ut = np.zeros(self.n_pool)
        self.vt = np.zeros(self.n_pool)
        self.it = np.zeros(self.n_pool)
        self.vit = np.zeros(self.n_pool)
        self.ucb_uncertain = np.zeros(self.n_pool) 
        self.ucb_inconsist = np.zeros(self.n_pool)

        self.clf =  deepcopy(self.net)
        # if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
        # find_unused_parameters=True,
        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr=self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0
        train_acc = 0.
        previous_loss = 0.

        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            train_data_labeled = self.handler(self.X[idxs_train],
                                              torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                              transform=transform)
            loader_tr_labeled = DataLoader(train_data_labeled,
                                           shuffle=True,
                                           pin_memory=True,
                                           # sampler = DistributedSampler(train_data),
                                           worker_init_fn=self.seed_worker,
                                           generator=self.g,
                                           **{'batch_size': 250, 'num_workers': 1})
        if idxs_unlabeled.shape[0] != 0:
            mean = self.args.normalize['mean']
            std = self.args.normalize['std']
            train_data_unlabeled = self.handler(self.X[idxs_unlabeled],
                                                torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                                                transform=TransformUDA(mean=mean, std=std, size=self.args.img_size,channels=self.args.channels))
            loader_tr_unlabeled = DataLoader(train_data_unlabeled,
                                             shuffle=True,
                                             pin_memory=True,
                                             # sampler = DistributedSampler(train_data),
                                             worker_init_fn=self.seed_worker,
                                             generator=self.g,
                                             **{'batch_size': int(2250 * batch_ratio), 'num_workers': 1})
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule,
                                                                self.args)

                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins,
                                                                     need_secs)

                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer)
                test_acc = self.predict(self.X_te, self.Y_te)

                # measure elapsed time
                epoch_time.update(time.time() - ts)
                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                  need_time, current_learning_rate
                                                                                  ) \
                      + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                              1. - recorder.max_accuracy(False)))

                recorder.update(epoch, train_los, train_acc, 0, test_acc)

                # The converge condition
                if abs(previous_loss - train_los) < 0.0005:
                    break
                else:
                    previous_loss = train_los
            if self.args.save_model:
                self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module
            # self.save_tta_values(self.get_tta_values())


        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc


   