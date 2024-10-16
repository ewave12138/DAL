import sys
import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import os
import torchvision
from torchvision.datasets import CIFAR100, CIFAR10, MNIST
from PIL import Image, ImageOps, ImageFilter

def get_dataset(name, path):
    if name.lower() == 'mnist':
        return get_MNIST(path)
    elif name.lower() == 'fashionmnist':
        return get_FashionMNIST(path)
    elif name.lower() == 'svhn':
        return get_SVHN(path)
    elif name.lower() == 'cifar10':
        return get_CIFAR10(path)
    elif name.lower() == 'cifar100':
        return get_CIFAR100(path)
    elif name.lower() == 'gtsrb':
        return get_GTSRB(path)
    elif name.lower() == 'tinyimagenet':
        return get_tinyImageNet(path)

def get_ImageNet(path):
    raw_tr = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/train')
    imagenet_tr_path = path +'imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/'
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((64, 64))])
    imagenet_folder = datasets.ImageFolder(imagenet_tr_path, transform=transform)
    idx_to_class = {}
    for (class_num, idx) in imagenet_folder.class_to_idx.items():
        idx_to_class[idx] = class_num
    X_tr,Y_tr = [], []
    item_list = imagenet_folder.imgs
    for (class_num, idx) in raw_tr.class_to_idx.items():
        new_img_num = 0
        for ii, (path, target) in enumerate(item_list):
            if idx_to_class[target] == class_num:
                X_tr.append(np.array(imagenet_folder[ii][0]))
                Y_tr.append(idx)
                new_img_num += 1
            if new_img_num >= 250:
                break
            
    return np.array(X_tr), np.array(Y_tr)


def get_tinyImageNet(path):
    # 100000 train 10000 test
    raw_tr = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/train')
    raw_te = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/val')
    f = open(path + '/tinyImageNet/tiny-imagenet-200/val/val_annotations.txt')

    val_dict = {}
    for line in f.readlines():
        val_dict[line.split()[0]] = raw_tr.class_to_idx[line.split()[1]]
    X_tr,Y_tr,X_te, Y_te = [],[],[],[]
    
    div_list = [len(raw_tr)*(x+1)//10 for x in range(10)] # can not load at once, memory limitation
    i=0
    for count in div_list:
        loop = count - i
        for j in range(loop):
            image,target = raw_tr[i]
            X_tr.append(np.array(image))
            Y_tr.append(target)
            i += 1

    for i in range(len(raw_te)):
        img, label = raw_te[i]
        img_pth = raw_te.imgs[i][0].split('/')[-1]
        X_te.append(np.array(img))
        Y_te.append(val_dict[img_pth])

    return X_tr,Y_tr,X_te, Y_te
    # torch.tensor(X_tr), torch.tensor(Y_tr), torch.tensor(X_te), torch.tensor(Y_te)
    
def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/mnist', train=True, download=True)
    raw_te = datasets.MNIST(path + '/mnist', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/fashionmnist', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/fashionmnist', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path, split='train', download=True)
    data_te = datasets.SVHN(path, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/cifar10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/cifar10', train=False, download=True)
    X_tr = data_tr.data
    # print(np.array(X_tr[0]).shape)
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100(path):
    data_tr = datasets.CIFAR100(path + '/cifar100', train=True, download=True)
    data_te = datasets.CIFAR100(path + '/cifar100', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_GTSRB(path):
    train_dir = os.path.join(path, 'gtsrb/train')
    test_dir = os.path.join(path, 'gtsrb/test')
    transform_tr = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.RandomCrop(size = 32, padding=4),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(), 
                                    torchvision.transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])
    transform_te = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor(), 
                                    torchvision.transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])])
    train_data = torchvision.datasets.ImageFolder(train_dir,transform_tr)
    test_data = torchvision.datasets.ImageFolder(test_dir,transform_te)

    # X_tr = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in train_data.samples])
    # Y_tr = torch.from_numpy(np.array(train_data.targets))
    # X_te = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in test_data.samples])
    # Y_te = torch.from_numpy(np.array(test_data.targets))

    X_tr = np.array([d[0].numpy() for d in train_data])  
    Y_tr = torch.from_numpy(np.array(train_data.targets))
    X_te = np.array([d[0].numpy() for d in test_data])  
    Y_te = torch.from_numpy(np.array(test_data.targets))

    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.lower() == 'mnist':
        return DataHandler1
    elif name.lower() == 'fashionmnist':
        return DataHandler1
    elif name.lower() == 'svhn':
        return DataHandler2
    elif name.lower() == 'cifar10':
        return DataHandler3
    elif name.lower() == 'cifar100':
        return DataHandler3
    elif name.lower() == 'tinyimagenet':
        return DataHandler3
    elif name.lower() == 'gtsrb':
        return DataHandler5
    else:
        return DataHandler4


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.numpy() if not isinstance(x, np.ndarray) else x
            x = Image.fromarray(x, mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
    
class DataHandler5(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform


    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
            
        if self.transform is not None:
            x = np.transpose(x, (1, 2, 0))
            mean = np.array([0.3337, 0.3064, 0.3171])
            std = np.array([0.2672, 0.2564, 0.2629])
            x = (x * std + mean)
            x = np.clip(x, 0, 1)
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

# handler for waal
def get_wa_handler(name):
    if name.lower() == 'fashionmnist':
        return  Wa_datahandler1
    elif name.lower() == 'svhn':
        return Wa_datahandler2
    elif name.lower() == 'cifar10':
        return  Wa_datahandler3
    elif name.lower() == 'cifar100':
        return  Wa_datahandler3
    elif name.lower() == 'tinyimagenet':
        return  Wa_datahandler3
    elif name.lower() == 'mnist':
        return Wa_datahandler1
    elif name.lower() == 'gtsrb':
        return Wa_datahandler3


class Wa_datahandler1(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:
            # print (x_1)
            x_1 = Image.fromarray(x_1, mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2, mode='L')
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2



class Wa_datahandler2(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class Wa_datahandler3(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2

# get_CIFAR10('./dataset')

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


        
def load_dataset_forMobyv2(dataset, path, args):
    if dataset == 'cifar10' :
        normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        IMG_SIZE = 32
    elif dataset == 'cifar100':
        normalize = T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        IMG_SIZE = 32
    elif dataset == 'tinyimagenet':
        normalize = T.Normalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        IMG_SIZE = 64
    else:
        normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        IMG_SIZE = 32
    # Weak augmentations
    cifar100_train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
       normalize
    ])

    cifar10_train_transform = T.Compose([
        T.RandomHorizontalFlip(0.5),
        T.RandomCrop(size=IMG_SIZE, padding=4),
        T.ToTensor(),
        normalize
    ])


    # Strong augmentations
    transform_2 = T.Compose([
        # T.Resize(IMG_SIZE, interpolation=_pil_interp('bicubic')),
        T.RandomResizedCrop(IMG_SIZE, scale=(0.2, 1.)),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur()], p=0.2),
        T.RandomApply([ImageOps.solarize], p=0.2),
        T.ToTensor(),
        normalize,
    ])

    cifar10_train_transform2 = transform_2 
    cifar100_train_transform2  = transform_2 
    
    # Test augmentations
    cifar100_test_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    cifar10_test_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])
    
    data_train2, data_unlabeled2,data_unlabeled = [], [],[]
    transform1 = T.Compose([T.ToTensor()])
    transform2 = T.Compose([T.ToTensor()])
    transform_test = T.Compose([T.ToTensor()])
    if dataset == 'cifar10': 
        data_unlabeled = Dataset_uniform(dataset, path , True, cifar10_train_transform)
        data_unlabeled2 = Dataset_uniform(dataset, path , True, cifar10_train_transform2)       # ulb traindata, using strong augmentation        
        transform1 = cifar10_train_transform
        transform2 = cifar10_train_transform2
        transform_test = cifar10_test_transform

    elif dataset == 'cifar100':
        data_unlabeled2 = Dataset_uniform(dataset, path ,True, cifar100_train_transform2)
        data_unlabeled = Dataset_uniform(dataset, path ,True, cifar100_train_transform)
        transform1 = cifar100_train_transform
        transform2 = cifar100_train_transform2
        transform_test = cifar100_test_transform

    elif dataset == 'mnist':
        data_unlabeled2 = Dataset_uniform(dataset, path ,True, cifar100_train_transform2)
        data_unlabeled = Dataset_uniform(dataset, path ,True, T.Compose([T.ToTensor()]))
        transform1 = T.Compose([T.ToTensor()])
        transform_test = T.Compose([T.ToTensor()])

    elif dataset == 'gtsrb':
        transform1 = T.Compose([T.ToTensor()])    
        transform2 =   args.transform_tr
        transform_test = args.transform_te
        data_unlabeled2 = Dataset_uniform(dataset, path ,True, transform2)
        data_unlabeled = Dataset_uniform(dataset, path ,True, transform1)
        
    elif dataset == 'tinyimagenet':
        transform1 = args.transform_tr
        transform2 = transform_2
        transform_test = args.transform_te
        data_unlabeled2 = Dataset_uniform(dataset, path ,True, transform2)
        data_unlabeled = Dataset_uniform(dataset, path ,True, transform1)

    return transform1, data_unlabeled, transform_test,  transform2, data_unlabeled2

class moby_handler(Dataset):
    def __init__(self,X,Y,transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
       
    def __getitem__(self, index) :
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            # if(self.dataset == 'gtsrb'):
            #     x = Image.fromarray(np.transpose((x, (1, 2, 0))))
            # else:
            #     x = Image.fromarray(x)
            x = Image.fromarray(x)
            x= self.transform(x)
        return x, y ,index
    
    def __len__(self):
        return len(self.X)
    


# for mobyv2al,typi, TCM
class Dataset_uniform(Dataset):
    def __init__(self, dataset_name, data_path,train_flag, transf, weak_strong_flag= False):
        self.dataset_name = dataset_name
        self.weak_strong_flag = weak_strong_flag
        if self.dataset_name == "cifar10":
            self.cifar10 = CIFAR10(data_path + '/cifar10', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "cifar100":
            self.cifar100 = CIFAR100(data_path + '/cifar100', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "mnist":
            self.mnist = MNIST(data_path + '/mnist', train=train_flag, 
                                    download=True, transform=transf)
        if self.dataset_name == "gtsrb":
            data_path = os.path.join(data_path, 'gtsrb', 'train' if train_flag else 'test')
            self.image_paths, self.labels = self.load_gtsrb(data_path)
        if self.dataset_name == 'tinyimagenet':
            # print('*****',data_path)
            # sys.exit()
            data_path = os.path.join(data_path, 'tinyImageNet/tiny-imagenet-200', 'train' if train_flag else 'val')
            self.image_paths, self.labels = self.load_tinyimagenet(data_path, train_flag)


            # self.X, self.Y, self.X_te, self.Y_te = get_tinyImageNet(data_path)
            # if train_flag:
            #     self.X = self.X
            #     self.Y = self.Y
            # else:
            #     self.X = self.X_te
            #     self.Y = self.Y_te

    def __getitem__(self, index):
        if self.dataset_name == "cifar10":
            image, target = self.cifar10[index]
        elif self.dataset_name == "cifar100":
            image, target = self.cifar100[index]
        elif self.dataset_name == "mnist":
            image, target = self.mnist[index]
        elif self.dataset_name == "gtsrb":
            img_path = self.image_paths[index]
            image = Image.open(img_path)
            transform = T.Compose([
            T.Resize((32, 32)),  
            T.ToTensor(),        
            ])
            image = transform(image)
            target = self.labels[index]
        elif self.dataset_name == 'tinyimagenet':
            img_path = self.image_paths[index]
            image = Image.open(img_path).convert('RGB')
            transform = T.Compose([  
            T.ToTensor(),        
            ])
            image = transform(image)
            target = self.labels[index] 

            # image = transform(self.X[index])
            # target = self.Y[index]
        if(self.weak_strong_flag):
            x_weak, x_strong = image
            return x_weak, x_strong, target, index
        else:
            return image, target, index

    def __len__(self):
        if self.dataset_name == "cifar10":
            return len(self.cifar10)
        elif self.dataset_name == "cifar100":
            return len(self.cifar100)
        elif self.dataset_name == "mnist":
            return len(self.mnist)
        elif self.dataset_name == "gtsrb":
            return len(self.image_paths)
        elif self.dataset_name == 'tinyimagenet':
            #return len(self.X)
            return len(self.image_paths)
        
    def load_gtsrb(self, path):
        image_paths = []
        labels = []
        for label in os.listdir(path):
            class_dir = os.path.join(path, label)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.ppm'):  # Assuming GTSRB images are in PPM format
                        img_path = os.path.join(class_dir, img_name)
                        image_paths.append(img_path)
                        labels.append(int(label))  # Assuming label is the directory name
        return image_paths, labels


    def load_tinyimagenet(self, path, split):
        image_paths = []
        labels = []
        wnids_path = os.path.join(os.path.dirname(path), 'wnids.txt')  
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f]
        if split :
            for i, wnid in enumerate(wnids):
                class_dir = os.path.join(path, wnid, 'images')
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(i)

        else:
            val_annotations_path = os.path.join(path, 'val_annotations.txt')
            with open(val_annotations_path, 'r') as f:
                val_annotations = [line.strip().split() for line in f]

            for annotation in val_annotations:
                img_name = annotation[0]
                wnid = annotation[1]
                img_path = os.path.join(path, 'images', img_name)
                label = wnids.index(wnid) 
                image_paths.append(img_path)
                labels.append(label)

        return image_paths, labels


""" 
    AugmentedDataset
    Returns an image together with an augmentation.
    for TCM use
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target, _ = self.dataset.__getitem__(index)
        sample = {'image': self.image_transform(image), 
                  'target': target}
        sample['image_augmented'] = self.augmentation_transform(image)
        return sample