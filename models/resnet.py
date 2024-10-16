'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_fea(nn.Module):
    def __init__(self, block, num_blocks):
        super(resnet_fea, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, img_size, intermediate=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        avg_pool_size = img_size//8
        out = F.avg_pool2d(out4, avg_pool_size)
        out = out.view(out.size(0), -1)
        return out, [out1, out2, out3, out4]

class resnet_clf(nn.Module):
    def __init__(self, block, n_class=10):
        super(resnet_clf, self).__init__()
        self.linear = nn.Linear(128 * block.expansion, n_class)

    def forward(self, x):
        # emb = x.view(x.size(0), -1)
        out = self.linear(x)
        return out, x

class resnet_dis(nn.Module):
    def __init__(self, embDim):
        super(resnet_dis, self).__init__()
        self.dis_fc1 = nn.Linear(embDim, 50)
        self.dis_fc2 = nn.Linear(50, 1)

    def forward(self, x):
        e1 = F.relu(self.dis_fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.dis_fc2(x)
        x  = torch.sigmoid(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=10, bayesian=False):
        super(ResNet, self).__init__()
        # self.in_planes = 16
        self.embDim = 128 * block.expansion
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        # self.linear = nn.Linear(128 * block.expansion, n_class)

        # self.dis_fc1 = nn.Linear(512, 50)
        # self.dis_fc2 = nn.Linear(50, 1)

        self.feature_extractor = resnet_fea(block, num_blocks)
        self.linear = resnet_clf(block, n_class)
        self.discriminator = resnet_dis(self.embDim)
        self.bayesian = bayesian

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    # def feature_extractor(self, x): # feature extractor
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     emb = out.view(out.size(0), -1)
    #     return emb


    def forward(self, x, intermediate=False):
        out, in_values = self.feature_extractor(x, x.shape[2])
        # apply dropout to approximate the bayesian networks
        out = F.dropout(out, p=0.2, training=self.bayesian)
        # emb = emb.view(emb.size(0), -1)
        out, emb = self.linear(out)
        if intermediate == True:
            return out, emb, in_values
        else:
            return out, emb

    def get_embedding_dim(self):
        return self.embDim

def blur(in_filters, sfilter=(1, 1), pad_mode="constant"):
    if tuple(sfilter) == (1, 1) and pad_mode in ["constant", "zero"]:
        layer = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)
    else:
        layer = Blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)
    return layer

class Blur(nn.Module):

    def __init__(self, in_filters, sfilter=(1, 1), pad_mode="replicate", **kwargs):
        super(Blur, self).__init__()

        filter_size = len(sfilter)
        self.pad = SamePad(filter_size, pad_mode=pad_mode)

        self.filter_proto = torch.tensor(sfilter, dtype=torch.float, requires_grad=False)
        self.filter = torch.tensordot(self.filter_proto, self.filter_proto, dims=0)
        self.filter = self.filter / torch.sum(self.filter)
        self.filter = self.filter.repeat([in_filters, 1, 1, 1])
        self.filter = torch.nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filter, groups=x.size()[1])

        return x

    def extra_repr(self):
        return "pad=%s, filter_proto=%s" % (self.pad, self.filter_proto.tolist())


class SamePad(nn.Module):

    def __init__(self, filter_size, pad_mode="constant", **kwargs):
        super(SamePad, self).__init__()

        self.pad_size = [
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
            int((filter_size - 1) / 2.0), int(math.ceil((filter_size - 1) / 2.0)),
        ]
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.pad_size, mode=self.pad_mode)

        return x

    def extra_repr(self):
        return "pad_size=%s, pad_mode=%s" % (self.pad_size, self.pad_mode)

class TanhBlurBlock(nn.Module):

    def __init__(self, in_filters, temp=1e1, sfilter=(1, 1), pad_mode="constant", **kwargs):
        super(TanhBlurBlock, self).__init__()

        self.temp = temp
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.blur = blur(in_filters, sfilter=sfilter, pad_mode=pad_mode)

    def forward(self, x):
        x = self.temp * self.tanh(x / self.temp)
        x = self.relu(x)
        x = self.blur(x)

        return x

    def extra_repr(self):
        return "temp=%.3e" % self.temp
    

class ResNetEnc(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, avg_pool=4,lenet_flag =False):
        super(ResNetEnc, self).__init__()
        self.in_planes = 64
        self.avg_pool = avg_pool
        self.num_features = 512
        self.lenet_flag=lenet_flag
        sblock=TanhBlurBlock 
        num_sblocks=(0, 0, 0, 0)
        if(lenet_flag):
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear = nn.Linear(512, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)
        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0])# **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1])# **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2])# **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3])# **block_kwargs)
        # self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4], **block_kwargs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def _make_layer(block, in_channels, out_channels, num_blocks, pool, **block_kwargs):
    #     layers, channels = [], in_channels
    #     if pool:
    #         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    #     for _ in range(num_blocks):
    #         layers.append(block(channels, out_channels, **block_kwargs))
    #         channels = out_channels
    #     return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out1 = self.smooth0(out1)

        out2 = self.layer2(out1)
        out2 = self.smooth1(out2)

        out3 = self.layer3(out2)
        out3 = self.smooth2(out3)

        if(self.lenet_flag):
            out = F.avg_pool2d(out3, self.avg_pool)
            
        else:
            out4 = self.layer4(out3)
            out4 = self.smooth3(out4)

            ###
            pool_size = out4.shape[2]
            out = F.avg_pool2d(out4, pool_size)
            ###
            
            #out = F.avg_pool2d(out4, self.avg_pool)
        # outf = out.view(out.size(0), -1)
        self.num_features = out.shape[1]
        # outl = self.linear(outf)
        # out = self.linear(outf)
        return out #, outf, [out1, out2, out3, out4]

def ResNet18E(num_classes = 10, avg_pool = 4, lenet_flag = False):
    return ResNetEnc(BasicBlock, [2,2,2,2], num_classes, avg_pool,lenet_flag)

def ResNet18(n_class, bayesian=False):
    return ResNet(BasicBlock, [2,2,2,2], n_class=n_class, bayesian=bayesian)

def ResNet34(n_class, bayesian=False):
    return ResNet(BasicBlock, [3,4,6,3], n_class=n_class, bayesian=bayesian)

def ResNet50(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,4,6,3], n_class=n_class, bayesian=bayesian)

def ResNet101(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,4,23,3], n_class=n_class, bayesian=bayesian)

def ResNet152(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,8,36,3], n_class=n_class, bayesian=bayesian)



class ResNetC(nn.Module):
    def __init__(self, in_channels, num_classes, **block_kwargs):
        super(ResNetC, self).__init__()

        self.linear = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


class MLPBlock(nn.Module):

    def __init__(self, in_features, num_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = nn.Linear(in_features, 4096, True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(4096, 4096, True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense3 = nn.Linear(4096, num_classes, True)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x


class VGGNet2C(nn.Module):
    def __init__(self, in_channels, num_classes, cblock=MLPBlock, **block_kwargs):
        super(VGGNet2C, self).__init__()

        self.classifier = []
        if cblock is MLPBlock:
            # self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
            self.classifier.append(cblock(7 * 7 * in_channels, num_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(in_channels, num_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
    
    def forward(self, x):
        x = self.classifier(x)
        return x


class VGGNet2Enc(nn.Module):

    def __init__(self, block, num_blocks, filter=3,
                 sblock=TanhBlurBlock, num_sblocks=(0, 0, 0, 0, 0),
                 cblock=MLPBlock,
                 num_classes=10, name="vgg", **block_kwargs):
        super(VGGNet2Enc, self).__init__()

        self.name = name
        self.num_features = 512
        self.layer0 = self._make_layer(block, filter, 64, num_blocks[0], pool=False, **block_kwargs)
        self.layer1 = self._make_layer(block, 64, 128, num_blocks[1], pool=True, **block_kwargs)
        self.layer2 = self._make_layer(block, 128, 256, num_blocks[2], pool=True, **block_kwargs)
        self.layer3 = self._make_layer(block, 256, 512, num_blocks[3], pool=True, **block_kwargs)
        self.layer4 = self._make_layer(block, 512, 512, num_blocks[4], pool=True, **block_kwargs)

        self.smooth0 = self._make_smooth_layer(sblock, 64, num_sblocks[0], **block_kwargs)
        self.smooth1 = self._make_smooth_layer(sblock, 128, num_sblocks[1], **block_kwargs)
        self.smooth2 = self._make_smooth_layer(sblock, 256, num_sblocks[2], **block_kwargs)
        self.smooth3 = self._make_smooth_layer(sblock, 512, num_sblocks[3], **block_kwargs)
        self.smooth4 = self._make_smooth_layer(sblock, 512, num_sblocks[4], **block_kwargs)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.classifier = []
        # if cblock is MLPBlock:
        #     # self.classifier.append(nn.MaxPool2d(kernel_size=2, stride=2))
        #     self.classifier.append(nn.AdaptiveAvgPool2d((7, 7)))
        #     self.classifier.append(cblock(7 * 7 * 512, num_classes, **block_kwargs))
        # else:
        #     self.classifier.append(cblock(512, num_classes, **block_kwargs))
        # self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(block, in_channels, out_channels, num_blocks, pool, **block_kwargs):
        layers, channels = [], in_channels
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(num_blocks):
            layers.append(block(channels, out_channels, **block_kwargs))
            channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_smooth_layer(sblock, in_filters, num_blocks, **block_kwargs):
        layers = []
        for _ in range(num_blocks):
            layers.append(sblock(in_filters=in_filters, **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer0(x)
        out1 = self.smooth0(out1)

        out2 = self.layer1(out1)
        out2 = self.smooth1(out2)

        out3 = self.layer2(out2)
        out3 = self.smooth2(out3)

        out4 = self.layer3(out3)
        out4 = self.smooth3(out4)
        if out4.shape[-1]==1:
            features = out4
        else:
            x = self.layer4(out4)
            features = self.smooth4(x)
            # features = x
            if features.shape[-1] != 1:
                features =nn.MaxPool2d(kernel_size=features.shape[-1], stride=2)(features)

        self.num_features = features.shape[1]
        # x = self.classifier(features)

        return features

def convnxn(in_channels, out_channels, kernel_size, stride=1, groups=1, padding=0):
        return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)


def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return convnxn(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, padding=1)


class vggBasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **block_kwargs):
        super(vggBasicBlock, self).__init__()

        self.conv = conv3x3(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def dnn_16enc(num_classes=10, filter=3, name="vgg_dnn_16", **block_kwargs):
    return VGGNet2Enc(vggBasicBlock, [2, 2, 3, 3, 3], filter,
                  num_classes=num_classes, name=name, **block_kwargs)

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
