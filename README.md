# Deep Active Learning with Pytorch
An implementation of the state-of-the-art Deep Active Learning algorithm. 
This code was built based on [Jordan Ash's repository](https://github.com/JordanAsh/badge).

# Dependencies

To run this code fully, you'll need [PyTorch](https://pytorch.org/) (we're using version 1.4.0), [scikit-learn](https://scikit-learn.org/stable/).
We've been running our code in Python 3.9.

# Algorithms Implemented
## Deep active learning Strategies
|                Sampling Strategies                |    Year    | Done |
|:-------------------------------------------------:|:----------:|:----:|
|                  Random Sampling                  |      x     |  ✅ |
|                 ClusterMargin [1]                 |  arXiv'21  |  ✅ |
|                      WAAL [2]                     | AISTATS'20 |  ✅ |
|                     BADGE [3]                     |   ICLR'20  |  ✅ |
|    Adversarial Sampling for Active Learning [4]   |   WACV'20  |  ✅ |
|       Learning Loss for Active Learning [5]       |   CVPR'19  |  ✅ |
|     Variational Adversial Active Learning [6]     |   ICCV'19  |  ✅ |
|                   BatchBALD [7]                   |   NIPS'19  |  ✅ |
|                K-Means Sampling [8]               |   ICLR'18  |  ✅ |
|                K-Centers Greedy [8]               |   ICLR'18  |  ✅ |
|                    Core-Set [8]                   |   ICLR'18  |  ✅ |
|             Adversarial - DeepFool [9]            |  ArXiv'18  |  ✅ |
|             Uncertainty Ensembles [10]            |   NIPS'17  |  ✅ |
| Uncertainty Sampling with Dropout Estimation [11] |   ICML'17  |  ✅ |
|     Bayesian Active Learning Disagreement [11]    |   ICML'17  |  ✅ |
|               Least Confidence [12]               |  IJCNN'14  |  ✅ |
|                Margin Sampling [12]               |  IJCNN'14  |  ✅ |
|               Entropy Sampling [12]               |  IJCNN'14  |  ✅ |
|               UncertainGCN Sampling [13]          |  CVPR'21  |  ✅ |
|               CoreGCN Sampling [13]               |  CVPR'21  |  ✅ |
|               Ensemble [14]                       |  CVPR'18  |  ✅ |
|               MCDAL [15]            |  Knowledge-based Systems'19  |  ✅ |
|               ALFA-Mix                            |  CVPR'22  |  ✅ |
|               TypiClust                           |  ICML'22  |  ✅ |
|               TCM                                 |  ICLR'24  |  ✅ |



## Deep active learning + Semi-supervised learning

|                Sampling Strategies                |    Year    | Done |
|:-------------------------------------------------:|:----------:|:----:|
|               Consistency-SSLAL [16]                |  ECCV'20  |  ✅ |
|               MixMatch-SSLAL [17]                |  arXiv  |  ✅ |
|               UDA [18]                |  NIPS'20  |  ✅ |
|               MoBYv2AL                            |  BMVC'22  |  ✅ |
|               ASSL                                |  ArXiv'23  |  ✅ |




# Running an experiment
## Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n DAL python=3.9
conda activate DAL
pip install -r requirements.txt
```

## Example
```
python main.py --model ResNet18  --dataset cifar10 --strategy LeastConfidence
```
It runs an active learning experiment using ResNet18 and CIFAR-10 data, querying according to the LeastConfidence algorithm. The result will be saved in the **./save** directory.

You can also use `run.sh` to run experiments.

## Self-supervised feautres of data
You can download the features/feature_model from [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155170454_link_cuhk_edu_hk/EtfELBvgwwlArSvrxtOHbaoBkTmCZgqZ3qOPwaQ601a4SQ?e=GfEj94)

# Contact
If you have any questions/suggestions, or would like to contribute to this repo, please feel free to contact:
  Yu Li `yuli@cse.cuhk.edu.hk`,   Muxi Chen `mxchen21@cse.cuhk.edu.hk` or Prof. Qiang Xu `qxu@cse.cuhk.edu.hk`

  

## References

[1] (ClusterMargin, 2021) Batch Active Learning at Scale

[2] (WAAL, AISTATS'20) Deep Active Learning: Unified and Principled Method for Query and Training [paper](https://arxiv.org/pdf/1911.09162.pdf) [code](https://github.com/cjshui/WAAL)

[3] (BADGE, ICLR'20) Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds [paper](https://openreview.net/forum?id=ryghZJBKPS) [code](https://github.com/JordanAsh/badge)

<!-- - [x] (PROXY, ICLR'20) Selection via Proxy: Efficient Data Selection for Deep Learning [paper](https://arxiv.org/pdf/1906.11829.pdf) [code](https://github.com/stanford-futuredata/selection-via-proxy) -->


[4] (ASAL, WACV'20) Adversarial Sampling for Active Learning [paper](https://arxiv.org/pdf/1808.06671.pdf) 

[5] (CVPR'19) Learning Loss for Active Learning [paper](https://arxiv.org/pdf/1905.03677v1.pdf) [code](https://github.com/Mephisto405/Learning-Loss-for-Active-Learning)

[6] (VAAL, ICCV'19) Variational Adversial Active Learning [paper](https://arxiv.org/pdf/1904.00370.pdf) [code](https://github.com/sinhasam/vaal)

[7] (BatchBALD, NIPS'19) BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning [paper](https://papers.nips.cc/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf) [code](https://github.com/BlackHC/BatchBALD)

<!-- - [Muxi] (ICML'19) Bayesian Generative Active Deep Learning [paper](https://arxiv.org/pdf/1904.11643v1.pdf) [code](https://github.com/toantm/BGADL) -->
<!-- - [YuLi] (AAAI'19) (SPAL) Self-Paced Active Learning: Query the Right Thing at the Right Time [paper](https://ojs.aaai.org//index.php/AAAI/article/view/4445)  -->

[8] (CORE-SET, ICLR'18) Active Learning for Convolutional Neural Networks: A Core-Set Approach [paper](https://arxiv.org/pdf/1708.00489.pdf) [code](https://github.com/ozansener/active_learning_coreset)

[9] (DFAL, 2018) Adversarial Active Learning for Deep Networks: a Margin Based Approach

[10] (NIPS'17) Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles [paper](https://arxiv.org/pdf/1612.01474.pdf) [code](https://github.com/vvanirudh/deep-ensembles-uncertainty) 

[11] (DBAL, ICML'17) Deep Bayesian Active Learning with Image Data [paper](https://arxiv.org/pdf/1703.02910.pdf) [code](https://github.com/bnjasim/Deep-Bayesian-Active-Learning)

[12] (Least Confidence/Margin/Entropy, IJCNN'14) A New Active Labeling Method for Deep Learning, IJCNN, 2014


<!-- - [x] (ICDM'20) Active Learning with Multi-granular Graph Auto-Encoder [paper](https://ieeexplore.ieee.org/document/9338373/authors#authors)  -->


[13] (UncertainGCN, CoreGCN, CVPR'21) Sequential Graph Convolutional Network for Active Learning [paper](https://arxiv.org/pdf/2006.10219.pdf) [code](https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning)

[14] (Emsemble, CVPR'18) The power of ensembles for active learning in image classification [paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf) 

[15] (Knowledge-based Systems'19) Multi-criteria active deep learning for image classification [paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705119300747?via%3Dihub) [code](https://github.com/houxingxing/Multi-Criteria-Active-Deep-Learning-for-Image-Classification)

[16] (ECCV'20) Consistency-based semi-supervised active learning: Towards minimizing labeling cost [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550511.pdf) 

[17] (Google, arXiv) Combining MixMatch and Active Learning for Better Accuracy with Fewer Labels 

[18] (Google, NIPS’20) Unsupervised Data Augmentation for Consistency Training 

[19] (ALFA-mix, CVPR'22) Active Learning by Feature Mixing

[20] (MoBYv2AL, BMVC'23) MoBYv2AL: Self-supervised Active Learning for Image Classification

[21] (TCM, ICLR'24) Bridging Diversity and Uncertainty in Active learning with Self-Supervised Pre-Training

[22] (Typiclust, ICML'22) Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets

[23] (ASSL, ArXiv'23) Active Semi-Supervised Learning by Exploring Per-Sample Uncertainty and Consistency

<!-- - [YuLi] (LAL, NIPS'17) Learning Active Learning from Data [paper](https://papers.nips.cc/paper/2017/file/8ca8da41fe1ebc8d3ca31dc14f5fc56c-Paper.pdf) [code](https://github.com/ksenia-konyushkova/LAL) -->

