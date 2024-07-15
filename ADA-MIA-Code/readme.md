<meta name="robots" content="noindex">
# ADA-MIA
## About The Project
ADA-MIA allows an adversary to conduct membership inference attack against contrastive pre-trained encoders via aggressive data augmentations with almost no priori knowledge.

The main function is contained in **Data_Augmentation.py**, **Gen_Mem_Features.py**, and **DC_Inference.py**.

## Getting Started
### Prerequisites
**ADA-MIA** requires the following packages: 
- Python 3.8.17
- torch 1.8.1+cu101
- torchvision 0.9.1+cu101
- Scikit-learn 1.3.0
- Numpy 1.24.4
- Scipy 1.10.1
- xlsxwriter 3.1.2
- opacus 0.13.0
- xgboost 1.7.6
- tensorboard 2.13.0
- pandas 2.0.3

### File Structure 
```
ADA-MIA
├── datasets
│   ├── CIFAR-10
│   ├── CIFAR-100
│   ├── STL-10
│   └── Tiny-ImageNet-200
├── MoCo_Pretrain.py
├── SimCLR_Pretrain.py
├── Data_Augmentation.py
├── Gen_Mem_Features.py
├── DC_Inference.py
├── ADAMIA_Sup.py
```

There are several parts of the code:
- datasets folder: This folder contains the training and testing data for the target model.  In order to reduce the memory space, we just list the  links to these dataset here. 
   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   - CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
   - STL-10: http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
   - Tiny-ImageNet-200: http://cs231n.stanford.edu/tiny-imagenet-200.zip

- MoCo-Pretrain.py: This file contains the pre-training of contrastive encoders using MoCo.
- MoCo-Pretrain.py: This file contains the pre-training of contrastive encoders using SimCLR.
- Data_Augmentation.py: This file contains the generation of data augmentations from weak to strong, which corresponds to the **Section 4.2** in our paper.
- Gen_Mem_Features.py: This file contains the process of membership feature generation.
- DC_Inference.py: This file contains the unsupervised inference via Deep-Cluster, which corresponds to the **Section 4.3** in our paper.
- ADAMIA_Sup.py: This file contains a rough example of the supervised version of our ADA-MIA, in case of the adversary having access to some records from the same distribution as pre-training dataset.

## Parameter Setting of ADA-MIA
The attack settings of ADA-MIA are determined in the parameters of **Data_Augmentation.py** in **DC_Inference.py**.
- ***Progressive Data Augmentation Settings***
-- max: the maximum strength of data augmentation
-- interval: the interval of data augmentation, which correspond to the parameter $\lambda$.
- ***Deep-Cluster training Settings***
-- inference_epochs: the num of training epoch of Deep-Cluster
-- inference_rounds: the num of rounds to conduct unsupervised inference
-- hidden_neurons: the num of hidden neurons within MLP in Deep-Cluster
-- output_dim: the dimension of the output of MLP in Deep-Cluster
-- learning_rate: the lr when training MLP in Deep-Cluster
-- weight_decay: the weight decay when training MLP in Deep-Cluster
-- batch_size: the bs when conducting inference

## Execute ADA-MIA
1. Run **MoCo_Pretrain.py** or **SimCLR_Pretrain.py** to get target encoders.
2. Run **Gen_Mem_Features.py** to obtain membership features for images to be inferred.
3. Run **DC_Inference.py** to conduct inference with Deep-Cluster on these images.
4. Run **ADAMIA_Sup.py** to conduct inference in supervised manner.
