from copy import deepcopy
import random
import sys
from tqdm import tqdm
from .strategy import Strategy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from .sampler_util import SubsetRandomSampler
from models import resnet,moby
from torch.utils.data import DataLoader
import os
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import abc
import numpy as np

class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def select_batch_unc_(self, **kwargs):
      return self.select_batch_unc_(**kwargs)

  def to_dict(self):
    return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, X,  metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
          self.min_distances = None
        if only_new:
          cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if cluster_centers:
          x = self.features[cluster_centers]
          # Update min_distances for all examples given new cluster center.
          dist = pairwise_distances(self.features, x, metric=self.metric)#,n_jobs=4)

          if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
          else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
          if self.already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)
          # New examples should not be in already selected since those points
          # should have min_distance of zero to a cluster center.
        #   assert ind not in already_selected

          self.update_distances([ind], only_new=True, reset_dist=False)
          new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))


        self.already_selected = already_selected

        return new_batch


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

class mobyv2al(Strategy):
	def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
		super(mobyv2al, self).__init__( X, Y, X_te, Y_te, idxs_lb, net, handler,args)
		self.n_pool  = len(Y)
		self.query_num = int(self.args.nQuery*self.n_pool/100)
		self.SUBSET = 10000
		
	def train_epoch_ssl(self, models, criterion, optimizers, dataloaders, 
										epoch, last_inter,schedulers):
		models['backbone'].train()
		models['classifier'].train()
		# TRAIN_CLIP_GRAD = True
		idx = 0
		num_steps = len(dataloaders['train'])
		c_loss_gain = 0.5 #- 0.05*cycle
		
		for (samples,samples_a) in zip(dataloaders['train'],dataloaders['train2']):
			
			samples_a = samples_a[0].cuda(non_blocking=True)
			samples_r = samples[0].cuda(non_blocking=True)
			targets   = samples[1].cuda(non_blocking=True)

			contrastive_loss, features, _ = models['backbone'](samples_a, samples_r, targets)

			if (idx % 2 ==0) or (idx <= last_inter):
				scores = models['classifier'](features)
				target_loss = criterion(scores, targets)
				t_loss = (torch.sum(target_loss)) / target_loss.size(0)
				c_loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
				loss = t_loss + c_loss_gain*c_loss
				# loss.backward()
			else:
				loss = c_loss_gain *(torch.sum(contrastive_loss)) / contrastive_loss.size(0)
			optimizers['backbone'].zero_grad()
			loss.backward()
			optimizers['backbone'].step()
			if (idx % 2 ==0) or (idx <= last_inter):
				optimizers['classifier'].zero_grad()
				optimizers['classifier'].step()
			if(idx % 10 == 0):
				print ("[Batch={:03d}] [Loss={:.2f}]".format(idx, loss))
			idx = idx + 1	
		return loss

	def test_without_ssl(self,models, epoch, no_classes, dataloaders, mode='test'):
		assert mode == 'val' or mode == 'test'
		models['backbone'].eval()
		models['classifier'].eval()
		total = 0
		correct = 0
		with torch.no_grad():
			total_loss = 0
			for (inputs, labels,*_) in dataloaders[mode]:
				
				inputs = inputs.cuda()
				labels = labels.cuda()

				_, feat, _ = models['backbone'](inputs,inputs,labels)
				# feat = models_b(inputs)
				scores = models['classifier'](feat)
				_, preds = torch.max(scores.data, 1)
				total += labels.size(0)
				correct += (preds == labels).sum().item()
		return correct / total

	
	def train_with_ssl(self,models,criterion,optimizers,schedulers,dataloaders,n_epoch,n_class,labeled_data,unlabeled_data,data_train,last_interleaved,query_num,dim_latent):
		print('train the model...')
		best_acc= 0.
		arg=0

		recorder = RecorderMeter(n_epoch)
		print('epochs:{}'.format(n_epoch))
		print('query_num = {}'.format(query_num))

		for epoch in range(n_epoch):
			loss = self.train_epoch_ssl(models,criterion,optimizers,dataloaders,n_epoch,last_interleaved,schedulers)
			
			#print('current_learning_rate = {}'.format(optimizers['backbone'].param_groups[0]['lr']))
			schedulers['classifier'].step(loss)
			schedulers['backbone'].step(loss)

			adjust_learning_rate(optimizers['backbone'], epoch, self.args.gammas, self.args.schedule, self.args)
			adjust_learning_rate(optimizers['classifier'], epoch, self.args.gammas, self.args.schedule, self.args)

			acc= self.test_without_ssl(models,n_epoch,n_class,dataloaders,mode='test')
			#print('>>>test_without_ssl end, acc={}\n'.format(acc))

			print('\n==>> [Epoch={:03d}/{:03d}]  [LR={:6.4f}]'.format( epoch, n_epoch, optimizers['classifier'].param_groups[0]['lr']) \
                + ' [Best : Test Accuracy={:.2f}]'.format(recorder.max_accuracy(False)))
			
			recorder.update(epoch, loss, 0, 0, acc)
			if best_acc < acc:
				best_acc = acc
		recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))	

		batchsize = self.args.loader_tr_args.get('batch_size')	
		models['classifier'].eval()
		models['backbone'].eval()
		features = np.empty((batchsize, dim_latent))
		k_var = 2
		c_loss_m = np.zeros((k_var, batchsize*len(dataloaders['unlabeled'])))
		lenet_flag = True if self.args.model == 'LeNet' else False
		models_b = resnet.ResNet18E(num_classes=n_class,lenet_flag=lenet_flag ).cuda()
		models_b.eval()
		combined_dataset = DataLoader(data_train, batch_size=batchsize, 
                                    sampler=SubsetSequentialSampler(np.concatenate((unlabeled_data,labeled_data))), 
                                    pin_memory=True, drop_last=False)
		for ulab_data in combined_dataset:        
			unlab = ulab_data[0].cuda()
			# target = ulab_data[1].cuda()
			feat =  models_b(unlab)
			feat = feat.detach().cpu().numpy()
			feat = np.squeeze(feat)

			
			features = np.concatenate((features, feat), 0)

		features = features[batchsize:,:]
		subset = len(unlabeled_data)
		labeled_data_size = len(labeled_data)
		# Apply CoreSet for selection
		new_av_idx = np.arange(subset,(subset + labeled_data_size))
		sampling = kCenterGreedy(features)  
		batch = sampling.select_batch_(new_av_idx, query_num)
			# print(min(batch), max(batch))
		other_idx = [x for x in range(subset) if x not in batch]
		arg = np.array(other_idx + batch)

		return best_acc, arg
	

	def train(self, alpha=0.1, n_epoch=10):
		
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		transform1, data_unlabeled, transform_test,  transform2, data_unlabeled2 = dataset.load_dataset_forMobyv2(self.args.dataset,self.args.data_path,self.args)
		#data_train, data_unlabeled, data_test, no_train, data_train2, data_unlabeled2 = dataset.load_dataset(self.args.dataset,self.args.data_path)
		# if self.args.dataset in ["cifar10", "cifar100"]:
		# 	#use the moby augmentation
		# 	data_train = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform1)
		# 	data_train2 = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform2)
		# 	data_test = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform_test) 
		# elif self.args.dataset == 'gtsrb':
		# 	#use our augmentation
		# 	data_train = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform1)
		# 	data_train2 = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform2)
		# 	data_test = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform_test)
		# elif self.args.dataset == 'tinyimagenet':
		data_train = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform1)
		data_train2 = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , True,transform2)
		data_test = dataset.Dataset_uniform(self.args.dataset,self.args.data_path , False,transform_test)
		#n_pool: len of train set
		#index of label data, ulb
		idxs_lb_train = np.arange(self.n_pool)[self.idxs_lb]
		idx_ulb_train = np.arange(self.n_pool)[~self.idxs_lb]
		random.shuffle(idx_ulb_train)
		
		print('len of data_labelled:{}\n'.format(len(idxs_lb_train)))
		
		# data_train = dataset.moby_handler(self.X,self.Y,transform1)
		# data_train2 = dataset.moby_handler(self.X,self.Y,transform2)
		# data_test = dataset.moby_handler(self.X_te,self.Y_te,transform_test) 
		
		len_labeled= len(idxs_lb_train)
		tr_batch = self.args.loader_tr_args.get('batch_size')
		te_batch = self.args.loader_te_args.get('batch_size')
		if self.SUBSET < len_labeled:
			self.SUBSET = len_labeled
		k = int(self.SUBSET/tr_batch)
		self.SUBSET = k * tr_batch
		subset = idx_ulb_train[:self.SUBSET]
		  
		
		# interleave labelled and unlabelled batches
		if len(subset)> len_labeled :
			interleaved_size = 2 * int(len_labeled/tr_batch) * tr_batch
		else:
			interleaved_size = 2 * int(len(subset)/tr_batch) * tr_batch

		interleaved = np.zeros((interleaved_size)).astype(int)
		if len_labeled > len(subset):
			l_mixed_set = len(subset)

		else:
			l_mixed_set = len_labeled 
		for cnt in range(2*int(l_mixed_set/tr_batch)):
			idx = int(cnt / 2)
			if cnt % 2 == 0:
				interleaved[cnt*tr_batch:(cnt+1)*tr_batch] = idxs_lb_train[idx*tr_batch:(idx+1)*tr_batch]                             
			else:
				interleaved[cnt*tr_batch:(cnt+1)*tr_batch] = subset[idx*tr_batch:(idx+1)*tr_batch] 
		interleaved = interleaved.tolist()
		last_interleaved = idx
	
		test_loader = DataLoader(data_test,batch_size=te_batch,drop_last=True)
		lab_loader = DataLoader(data_train, batch_size=tr_batch, 
								sampler=SubsetSequentialSampler(interleaved), 
								pin_memory=True,drop_last=True)
		lab_loader2 = DataLoader(data_train2, batch_size=tr_batch, 
								sampler=SubsetSequentialSampler(interleaved), 
								pin_memory=True,drop_last=True)
		unlab_loader2 = DataLoader(data_unlabeled2, batch_size=tr_batch, 
								sampler=SubsetSequentialSampler(subset), 
								pin_memory=True,drop_last=True)
		unlab_loader = DataLoader(data_unlabeled, batch_size=tr_batch, 
								sampler=SubsetSequentialSampler(subset), 
								pin_memory=True,drop_last=True)
		dataloaders  = {'train': lab_loader, 'train2': lab_loader2, 
						'test': test_loader, 'unlabeled': unlab_loader, 'unlabeled2': unlab_loader2}

		

		if(self.args.model == 'LeNet'):
			dim_latent = 256
		else:
			dim_latent =512
		with torch.cuda.device(0):	
			classifier =resnet.ResNetC(dim_latent,num_classes=self.args.n_class).cuda()
			model = moby.build_MoBymodel(n_class=self.args.n_class, batch_size=tr_batch,no_of_labelled=interleaved,epochs=self.args.n_epoch,model_encoder=self.args.model).cuda()
		
		torch.backends.cudnn.benchmark = True
		models = {'backbone': model, 'classifier': classifier}
		Nes_flag = False
		criterion = nn.CrossEntropyLoss(reduction='none')
		optim_backbone = optim.SGD(models['backbone'].parameters(), lr=self.args.lr, 
               momentum=self.args.momentum, weight_decay=5e-4, nesterov=Nes_flag)
		optim_classifier = optim.SGD(models['classifier'].parameters(), lr=self.args.lr, 
            	momentum=self.args.momentum, weight_decay=0.0, nesterov=Nes_flag)
		
		sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=self.args.schedule,gamma=0.1)
		sched_classifier = lr_scheduler.MultiStepLR(optim_classifier, milestones=self.args.schedule,gamma=0.1)
		optimizers = {'backbone': optim_backbone, 'classifier': optim_classifier}
		schedulers = {'backbone': sched_backbone, 'classifier': sched_classifier} 

		#training and test
		acc,arg= self.train_with_ssl(models,criterion,optimizers,schedulers,dataloaders,self.args.n_epoch,
							 self.args.n_class,idxs_lb_train,subset,data_train,
							 last_interleaved,self.query_num,dim_latent)
		self.arg = arg
		self.idxs_lb = idxs_lb_train
		self.subset = subset
		self.dataloaders = dataloaders
		self.data_train = data_train
		return acc 

	def query(self,n):
		# self.idxs_lb += list(torch.tensor(self.subset)[self.arg][-self.query:].numpy())
		# self.dataloaders['train'] = DataLoader(self.data_train, batch_size=self.batchsize, 
        #                                 sampler=SubsetRandomSampler(self.idxs_lb), 
        #                                 pin_memory=True)
		valid_indices = [idx for idx in self.arg if idx < len(self.subset)]
		if len(valid_indices) < n:
			return torch.tensor(self.subset)[valid_indices].numpy()
		else:
        # 否则，返回最后n个有效索引对应的元素
			return torch.tensor(self.subset)[valid_indices[-n:]].numpy()
		# return torch.tensor(self.subset)[self.arg][-n:].numpy()

