import torch
import argparse
import numpy as np
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from DataProcess import cut_dataset, cut_dataset_random
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.nn import PairwiseDistance
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
import xlsxwriter
import warnings
from TinyImageNet import TinyImageNet
import seaborn as sns
from Data_Augmentation import trans


def parse_args():
    parser = argparse.ArgumentParser(description="Generating membership features and inference dataset")

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # ======================== Load Target Dataset =========================
    parser.add_argument('--target_dataset_path', type=str, default='./datasets')

    # ======================== Load Target Encoder =========================
    parser.add_argument('--target_encoder_path', type=str, default='./PPretrained_Encoders/'
                                                                   'mocov3-cifar10-3gpr99oc-ep=999.ckpt')

    # ======================== Membership Features Generation Configs =========================
    parser.add_argument('--inference_num', type=int, default=50, help="how many images to be inferred")

    return parser.parse_args()


args = parse_args()
warnings.filterwarnings('ignore')
p_dist = nn.PairwiseDistance(p=2)
c_sim = CosineSimilarity(dim=1)

print(f'generating membership features on: {args.device}')

# use pre-trained ResNet18 and load state_dict
backbone = resnet18()
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
backbone.maxpool = nn.Identity()
backbone.fc = nn.Identity()
state = torch.load(args.target_encoder_path, map_location='cpu')["state_dict"]
for j in list(state.keys()):
    if "backbone" in j:
        state[j.replace("backbone.", "")] = state[j]
    del state[j]
backbone.load_state_dict(state, strict=False)
backbone = backbone.to(args.device)


def train(num=100):
    train = CIFAR10(root=args.target_dataset_path, train=True, transform=trans(), download=False)
    # train = TinyImageNet(root='./data/tiny-imagenet-200', train=True, transform=trans())
    train1 = cut_dataset(train, 10000)
    with torch.no_grad():
        bank = torch.empty(num, len(train[0][0]) - 1, device=args.device)
        for i in tqdm(range(0, num), desc='extracting training samples'):
            data, sim = torch.empty(len(train[0][0]), 3, 32, 32, device=args.device), torch.empty(0, device=args.device)
            features = torch.empty(len(train[0][0]), 1, 512, device=args.device)
            train2 = cut_dataset_random(train1, 1)
            train_loader = DataLoader(train2, batch_size=1, num_workers=0)
            for inputs, _ in train_loader:
                # print()
                features[0] = backbone(inputs[0].to(args.device))
                for j in range(1, len(train[0][0])):
                    data[j] = inputs[j].to(args.device)
                    features[j] = backbone(data[j].unsqueeze(0))
                    sim = torch.cat((sim, c_sim(features[0], features[j])))
                bank[i] = sim
                sim = torch.empty(0, device=args.device)
            del data, features
    # print(bank)
    # return np.round(bank.cpu(), 20)
    return bank


def test(num=100):
    test = CIFAR10(root=args.target_dataset_path, train=False, transform=trans(), download=False)
    # test = TinyImageNet(root='./data/tiny-imagenet-200', train=False, transform=trans())
    test1 = cut_dataset(test, 10000)
    with torch.no_grad():
        bank = torch.empty(num, len(test[0][0]) - 1, device=args.device)
        for i in tqdm(range(0, num), desc='extracting testing samples'):
            data, sim = torch.empty(len(test[0][0]), 3, 32, 32, device=args.device), torch.empty(0, device=args.device)
            features = torch.empty(len(test[0][0]), 1, 512, device=args.device)
            test2 = cut_dataset_random(test1, 1)
            test_loader = DataLoader(test2, batch_size=1, num_workers=0)
            for inputs, _ in test_loader:
                # print()
                features[0] = backbone(inputs[0].to(args.device))
                for j in range(1, len(test[0][0])):
                    data[j] = inputs[j].to(args.device)
                    features[j] = backbone(data[j].unsqueeze(0))
                    sim = torch.cat((sim, c_sim(features[0], features[j])))
                bank[i] = sim
                sim = torch.empty(0, device=args.device)
            del data, features
    # print(bank)
    # return np.round(bank.cpu(), 20)
    return bank


def get_csv(num1=100, num2=100, root='./Logging/InferenceData.xlsx'):
    workbook = xlsxwriter.Workbook(root)
    worksheet = workbook.add_worksheet('new')
    member = train(num1).cpu()
    torch.save(member, './Logging/member.pt')
    print(f'the length of membership feature vector is: {member.size()[1]}')
    nonmem = test(num2).cpu()
    torch.save(nonmem, './Logging/nonmem.pt')

    for i in tqdm(range(0, num1), desc='getting csv for members'):
        for j in range(0, member.size()[1]):
            worksheet.write(i, j, member[i][j])
        worksheet.write(i, member.size()[1], 1)
    for ii in tqdm(range(0, num2), desc='getting csv for non-members'):
        for jj in range(0, member.size()[1]):
            worksheet.write(ii + num1, jj, nonmem[ii][jj])
        worksheet.write(ii + num1, member.size()[1], 0)
    workbook.close()
    ex = pd.read_excel('./Logging/GenInferenceData.xlsx')
    ex.to_csv('./Logging/InferenceData.csv', encoding='utf-8', sep=',', index=False)


def main():
    get_csv(int(args.inference_num / 2), int(args.inference_num / 2))


if __name__ == "__main__":
    main()
