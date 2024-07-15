import torch.nn as nn
from DeepCluster.Config import parse_args

args = parse_args()


class MLP_CE_EncoderMI(nn.Module):
    def __init__(self):
        super(MLP_CE_EncoderMI, self).__init__()
        self.BatchNorm1d_1 = nn.BatchNorm1d(256)
        self.BatchNorm1d_1.cuda()
        self.BatchNorm1d_2 = nn.BatchNorm1d(16)
        self.BatchNorm1d_2.cuda()
        self.MLP1 = nn.Sequential(
            nn.Linear(45, 256),
            self.BatchNorm1d_1,
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            self.BatchNorm1d_2,
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        classification = self.MLP1(x)
        return classification


class MLP_CE_DeepCluster(nn.Module):
    def __init__(self):
        super(MLP_CE_DeepCluster, self).__init__()
        # self.BatchNorm1d_1 = nn.BatchNorm1d(128)
        # self.BatchNorm1d_1.cuda()
        # self.BatchNorm1d_2 = nn.BatchNorm1d(32)
        # self.BatchNorm1d_2.cuda()
        self.MLP1 = nn.Sequential(
             nn.Linear(94, args.hidden_neurons),
             nn.ReLU(),
             # self.BatchNorm1d_1,
             nn.Linear(args.hidden_neurons, args.hidden_neurons),
             nn.ReLU(),
             nn.Linear(args.hidden_neurons, args.output_dim)
         )

        self.MLP2 = nn.Sequential(
            nn.Linear(args.output_dim, args.hidden_neurons),
            nn.ReLU(),
            # self.BatchNorm1d_2,
            nn.Linear(args.hidden_neurons, 2),
        )

    def forward(self, x):
        feature = self.MLP1(x)
        classification = self.MLP2(feature)
        return feature, classification
