import os
import torch
import pickle
import numpy as np
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from Config import parse_args

args = parse_args()
BATCH_SIZE = args.batch_size
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# def compute_features(trainloader, model, N):
#     model.eval()
#     # for i, _ in trainloader:
#     #     print(model(i).detach().numpy())
#     # discard the label information in the dataloader
#     # model.eval()
#     for i, (input_tensor, _) in enumerate(trainloader):
#         print(i)
#         input_var = torch.autograd.Variable(input_tensor, requires_grad=True).cpu()
#         # print(input_var)
#         # aux = model(input_var).data.cpu().numpy()
#         aux = model(input_var).detach().numpy()
#         features = np.zeros((200, aux.shape[1])).astype('float32')
#         if i < len(trainloader) - 1:
#             features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = aux.astype('float32')
#         else:
#             # special treatment for final batch
#             features[i * BATCH_SIZE:] = aux.astype('float32')
#         print(features)
#         return features


def compute_features(trainloader, model):
    rows = int(args.inference_num / args.batch_size)
    features = torch.empty(rows, args.batch_size, args.output_dim)
    for i, (input_tensor, _) in enumerate(trainloader):
        input_tensor = input_tensor.to(device)
        # print(input_tensor.size())
        features[i], _ = model(input_tensor)
    return features.detach().numpy().reshape(-1, args.output_dim)


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_PIL(dataset):
    N = len(dataset)
    imgs = [[] for i in range(N)]
    for i in range(N):
        narray = dataset[i][0]
        label = dataset[i][1]
        img = transforms.ToPILImage()(dataset[i][0]).convert('L')
        imgs[i].append(img)
        imgs[i].append(label)
    return imgs


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)
