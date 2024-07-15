import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors



def preprocess_features(data, n_pca = 256):
    data = data.astype('float32')
    # Apply PCA-whitening
    pca = PCA(2, whiten = True)
    pca.fit(data)
    data = pca.transform(data)
    return data

# K-means as clustering algorithm
# def run_kmeans(x, num_cluster=2):
#     kmeans = KMeans(n_clusters=num_cluster, max_iter=1000, tol=1e-2).fit(x)
#     pseudolabels = kmeans.predict(x)
#     return pseudolabels


# Spectral as clustering algorithm
# def run_kmeans(x, num_cluster=2):
#     kmeans = SpectralClustering(n_clusters=num_cluster, assign_labels='kmeans', n_init=500, n_neighbors=5)
#     pseudolabels = kmeans.fit_predict(x)
#     return pseudolabels

# T-sne as clustering algorithm
def run_kmeans(x, num_cluster=2):
    kmeans_1 = TSNE(
        n_components=10,
        n_iter=400,
        learning_rate='auto',
        method='exact',
        perplexity=30,
    )
    kmeans_2 = TSNE(
        n_components=num_cluster,
        n_iter=800,
        n_iter_without_progress=100,
        learning_rate='auto',
        perplexity=10,
    )
    pseudolabels = kmeans_1.fit_transform(x)
    pseudolabels = kmeans_2.fit_transform(pseudolabels)
    pseudolabels = torch.tensor(pseudolabels).argmax(dim=1)
    return np.array(pseudolabels)

# Agglomerelative as clustering algorithm
# def run_kmeans(x, num_cluster=2):
#     kmeans = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='complete')
#     pseudolabels = kmeans.fit_predict(x)
#     return pseudolabels

#


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        # self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        # print(image_indexes)
        # print(pseudolabels)
        # print(dataset)
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            # path = dataset[idx]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        # print(images)
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        # img = pil_loader(path)
        img = path
        # if self.transform is not None:
        #     img = self.transform(path)
        # print(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)

def cluster_assign(images_lists, dataset):
    pseudolabels = []
    image_indexes = []
    for cluster, idx in enumerate(images_lists):
        image_indexes.extend(idx)
        pseudolabels.extend([cluster] * len(idx))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    # t = transforms.Compose([transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         normalize])
    return ReassignedDataset(image_indexes, pseudolabels, dataset), pseudolabels

class Kmeans:
    def __init__(self, k):
        self.k = k
    def cluster(self, data):
        # PCA-reducing, whitening
        # xb = preprocess_features(data)
        # cluster the data
        # print(data)
        I = run_kmeans(data, self.k)
        # print(I)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(I)):
            self.images_lists[I[i]].append(i)
        return self.images_lists        #images_lists -> [[2, 3, 4], [5, 6], [1], [0]]

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]
    # pseudolabels  -> [0, 0, 0, 1, 1, 2, 3]
    # image_indexes -> [2, 3, 4, 5, 6, 1, 0]
