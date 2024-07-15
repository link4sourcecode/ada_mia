import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Membership Inference Attack against Contrastive Pre-trained Encoder"
                                                 "via Aggressive Data Augmentation (ADA-MIA)")
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    # ========================= DeepCluster (DC) Training Configs ==========================
    parser.add_argument('--inference_epochs', type=int, default=5, help='train DC for epochs in each inference round')
    parser.add_argument('--inference_rounds', type=int, default=10, help="infer membership for how many rounds")
    parser.add_argument('--inference_num', type=int, default=50, help="how many images to be inferred")

    parser.add_argument('--hidden_neurons', default=256, type=int, help="hidden state for MLP used in DC")
    parser.add_argument('--output_dim', default=32, type=int, help="output dimension for MLP used in DC")
    parser.add_argument('--learning_rate', default=0.00075, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--batch_size", default=25, type=float, help="bs for training MLP used in DC")

    return parser.parse_args()
