import argparse
# from opacus import PrivacyEngine
# from opacus.dp_model_inspector import DPModelInspector
# from opacus.utils import module_modification
# from opacus.utils.uniform_sampler import UniformWithReplacementSampler
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataProcess import cut_dataset
from TinyImageNet import TinyImageNet

import Utils
from Model import Model
from torchvision.models import resnet18

iters = 0


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _ in train_bar:
        x_q, x_k = x_q.cuda(), x_k.cuda()
        _, query = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda())
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            # data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            data, target = data.cuda(), target.cuda()
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_labels = sim_labels.to(torch.int64)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device, dtype=torch.float64)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # data prepare for CIAFR10
    train_data = Utils.CIFAR10Pair(root='../datasets', train=True, transform=Utils.train_transform, download=False)
    train_data1 = cut_dataset(train_data, 10000)
    train_loader = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    memory_data = Utils.CIFAR10Pair(root='../datasets', train=True, transform=Utils.test_transform, download=False)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_data = Utils.CIFAR10Pair(root='../datasets', train=False, transform=Utils.test_transform, download=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # data prepare for STL10
    # train_data = utils.STL10Pair(root='./data', split='unlabeled', transform=utils.train_transform, download=False)
    # train_data1 = cut_dataset(train_data, 10000)
    # train_loader = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                           drop_last=True)
    # memory_data = utils.STL10Pair(root='./data', split='train', transform=utils.test_transform, download=False)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                            drop_last=True)
    # test_data = utils.STL10Pair(root='./data', split='test', transform=utils.test_transform, download=False)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          drop_last=True)

    # data prepare for CIFAR100
    # train_data = utils.CIFAR100Pair(root='./data', train=True, transform=utils.train_transform, download=False)
    # train_data1 = cut_dataset(train_data, 10000)
    # train_loader = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                           drop_last=True)
    # memory_data = utils.CIFAR100Pair(root='./data', train=True, transform=utils.test_transform, download=False)
    # # memory_data = cut_dataset(memory_data, 10000)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                            drop_last=True)
    # test_data = utils.CIFAR100Pair(root='./data', train=False, transform=utils.test_transform, download=False)
    # test_data1 = cut_dataset(test_data, 1000)
    # test_loader = DataLoader(test_data1, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          drop_last=True)

    # data prepare for Tiny Imagenet
    # train_data = TinyImageNet(root='./data/tiny-imagenet-200', train=True, transform=utils.train_transform)
    # train_data1 = cut_dataset(train_data, 10000)
    # train_loader = DataLoader(train_data1, batch_size=batch_size, shuffle=True, num_workers=0)
    # memory_data = TinyImageNet(root='./data/tiny-imagenet-200', train=True, transform=utils.test_transform)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_data = TinyImageNet(root='./data/tiny-imagenet-200', train=False, transform=utils.test_transform)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # model setup and optimizer config
    model_q = Model(feature_dim)
    # print(model_q)
    model_k = Model(feature_dim)
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=2e-3, weight_decay=1e-6)

    # sr = batch_size / len(train_data1)
    # train_loader = DataLoader(
    #     train_data1,
    #     num_workers=0,
    #     batch_sampler=UniformWithReplacementSampler(num_samples=len(train_data1), sample_rate=sr)
    # )

    # privacy_engine = PrivacyEngine(
    #     module=model_q,
    #     epochs=1,
    #     target_delta=1e-5,
    #     target_epsilon=50.0,
    #     max_grad_norm=1.2,
    #     sample_rate=4 * sr,
    # )

    # inspector = DPModelInspector()
    # model_q = module_modification.convert_batchnorm_modules(model_q)
    model_q = model_q.cuda()
    print(model_q)
    # model_k = module_modification.convert_batchnorm_modules(model_k)
    model_k = model_k.cuda()
    # print(inspector.validate(model_q))
    # print(inspector.validate(model_k))

    # c as num of train class
    c = 10
    # init memory queue as unit random vector ---> [M, D]
    # memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs)
    loss = 10.0
    # privacy_engine.attach(optimizer)
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, train_loader, optimizer)
        # results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test(model_q, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
        if train_loss < loss:
            # iters = 0
            loss = train_loss
            torch.save(model_q.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
#         else:
#             iters += 1
#
#         if iters >= 10:
#             break
#
# print('early-stopping finished')
