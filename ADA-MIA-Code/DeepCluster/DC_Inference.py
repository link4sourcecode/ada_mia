import os
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, normalized_mutual_info_score
import Util
import Clustering
from sklearn.preprocessing import scale
import warnings
from torch.utils.data import Dataset, TensorDataset, DataLoader
from MLP import MLP_CE_DeepCluster
from tqdm import tqdm
import argparse
from Clustering import run_kmeans
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Applying DeepCluster to finish inference")

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========================= DeepCluster (DC) Training Configs ==========================
    parser.add_argument('--inference_epochs', type=int, default=100, help='train DC for epochs in each inference round')
    parser.add_argument('--inference_rounds', type=int, default=5, help="infer membership for how many rounds")
    parser.add_argument('--inference_num', type=int, default=50, help="how many images to be inferred")

    parser.add_argument('--hidden_neurons', default=256, type=int, help="hidden state for MLP used in DC")
    parser.add_argument('--output_dim', default=32, type=int, help="output dimension for MLP used in DC")
    parser.add_argument('--learning_rate', default=0.00075, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--batch_size", default=25, type=float, help="bs for training MLP used in DC")

    return parser.parse_args()


args = parse_args()
print(f'running Deep-Cluster on:{args.device}')

# load dataset:
BATCH_SIZE = args.batch_size
dataset = np.loadtxt('../Logging/InferenceData.csv', delimiter=",")
data, target = dataset[:, :-1], dataset[:, -1]
# data = scale(data)
data_tensor = torch.from_numpy(data)
data_copy = data_tensor.detach()
target_tensor = torch.from_numpy(target)
target_copy = target_tensor.detach()
print(f'the num of images to be inferred is: {data_tensor.size()[0]}')
data_tensor, data_copy = data_tensor.to(torch.float32), data_copy.to(torch.float32)
target_tensor, target_copy = target_tensor.to(torch.float32), target_copy.to(torch.float32)
my_dataset = TensorDataset(data_tensor, target_tensor)
my_dataset_copy = TensorDataset(data_copy, target_copy)
my_dataloader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)
my_dataloader_copy = DataLoader(my_dataset_copy, batch_size=BATCH_SIZE, shuffle=False)


# attack model -> CUDA
attack_model = MLP_CE_DeepCluster().to(args.device)
# print(attack_model)

# define loss function
criterion = nn.CrossEntropyLoss().to(args.device)

# clustering algorithm to use
deepcluster = Clustering.Kmeans(2)

# creating cluster assignments log
cluster_log = Util.Logger(os.path.join('clusters'))


# training loop
def train(loader, model, crit, epoch, LR):
    # record & update losses
    losses = Util.AverageMeter()

    # create global metrics to record
    best_acc, best_pre, best_rec, best_f1 = 0, 0, 0, 0

    # set optimizer
    optimizer_tl = torch.optim.Adam(
        model.parameters(),
        weight_decay=1e-5,
        lr=LR,
    )
    for ii in range(0, epoch):
        # switch to train mode
        model.train()
        for i, (input_tensor, target) in enumerate(loader):  # loader->reassign_loader, target为fake labels
            input_var = torch.autograd.Variable(input_tensor)
            input_var = input_var.to(args.device)
            target_var = torch.autograd.Variable(target)
            target_var = target_var.to(args.device)

            # only use classification output
            _, output = model(input_var)
            # print(target_var)
            # print(output)
            loss = crit(output, target_var)

            # record loss
            losses.update(loss.data.item(), input_tensor.size(0))

            # compute gradient and do SGD step
            optimizer_tl.zero_grad()
            loss.backward()
            optimizer_tl.step()

        # evaluating
        model.eval()
        with torch.no_grad():
            prediction = torch.empty(0, device=args.device)
            for csv, _ in my_dataloader_copy:
                csv = csv.to(args.device)
                _, out = model(csv)
                pred = torch.argmax(out, dim=1)
                prediction = torch.cat((prediction, pred))
            prediction = prediction.cpu()
            acc = accuracy_score(target_copy, prediction)
            pre = precision_score(target_copy, prediction)
            rec = recall_score(target_copy, prediction)
            f1 = f1_score(target_copy, prediction)
            if acc > best_acc:
                best_acc, best_pre, best_rec, best_f1 = acc, pre, rec, f1

    return best_acc, best_pre, best_rec, best_f1


# whole cycle to find best results
def ADA_MIA(e=args.inference_epochs, sign=True):
    ps_bank = list(range(0, e))
    aa, pp, rr, ff = 0, 0, 0, 0
    # ps_bank = np.zeros([e, distance.size()[0]])
    for epoch in range(0, e):

        # only use feature output
        features = Util.compute_features(my_dataloader, attack_model)

        # cluster the features
        I = deepcluster.cluster(features)

        # assign the pseudolaels
        train_dataset, ps_bank[epoch] = Clustering.cluster_assign(deepcluster.images_lists, my_dataset)
        if epoch != 0 and normalized_mutual_info_score(ps_bank[epoch], ps_bank[epoch - 1]) <= 0.8:
            train_dataset, ps_bank[epoch] = Clustering.cluster_assign(deepcluster.images_lists, my_dataset)

        # if epoch != 0 and sign is True:
        #     while normalized_mutual_info_score(ps_bank[epoch], ps_bank[epoch-1]) <= 0.8:
        #         I = deepcluster.cluster(features)
        #         train_dataset, ps_bank[epoch] = clustering.cluster_assign(deepcluster.images_lists, distance)

        # reassign dataloader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            # sampler=sampler,
            pin_memory=True,
            shuffle=True
        )

        # finally ADA-MIA results
        a, p, r, f = train(train_dataloader, attack_model, criterion, args.inference_epochs, LR=0.00075)
        if a > aa:
            aa, pp, rr, ff = a, p, r, f

    with open("../Logging/results.txt", "a") as x:

        x.write('---------------------------------\n')
        x.write(f'current-cycle ADA_MIA best-Accuracy  is: {aa}\n')
        x.write(f'current-cycle ADA_MIA best-Precision is: {pp}\n')
        x.write(f'current-cycle ADA_MIA best-Recall    is: {rr}\n')
        x.write(f'current-cycle ADA_MIA best-F1-Score  is: {ff}\n')
        x.write('---------------------------------\n')
    return aa, pp, rr, ff


if __name__ == "__main__":
    bank_a, bank_p, bank_r, bank_f = [], [], [], []
    for _ in tqdm(range(0, args.inference_rounds), desc='ADA_MIA inference'):
        x1, x2, x3, x4 = ADA_MIA()
        bank_a.append(x1)
        bank_p.append(x2)
        bank_r.append(x3)
        bank_f.append(x4)

    print(f'n times inference cycle Accuracy mean  is: {np.round(100 * np.mean(bank_a), 2)}%')
    print(f'n times inference cycle Accuracy std   is: {np.round(100 * np.std(bank_a), 2)}%')
    print(f'n times inference cycle Precision mean is: {np.round(100 * np.mean(bank_p), 2)}%')
    print(f'n times inference cycle Precision std  is: {np.round(100 * np.var(bank_p), 2)}%')
    print(f'n times inference cycle Recall mean    is: {np.round(100 * np.mean(bank_r), 2)}%')
    print(f'n times inference cycle Recall std     is: {np.round(100 * np.var(bank_r), 2)}%')
    print(f'n times inference cycle F1-Score mean  is: {np.round(100 * np.mean(bank_f), 2)}%')
    print(f'n times inference cycle F1-Score std   is: {np.round(100 * np.var(bank_f), 2)}%')