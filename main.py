import argparse
import torch
from dataset.dataset import BeatSaberDataset
from model.fact_model import FACT
from train.train import train
import os
from evaluation.evaluation import evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Exp
    parser.add_argument('--exp_id', type=str, default='exp_0')
    # Data
    parser.add_argument('--map_path', type=str, default="./dataset/data/map_feature_60/")
    parser.add_argument('--difficulty_path', type=str, default="./dataset/data/difficulty_list/expert_plus_file.pkl")
    parser.add_argument('--audio_path', type=str, default="./dataset/data/audio_feature/")
    parser.add_argument('--bpm_path', type=str, default="./dataset/data/audio_tempo.json")
    parser.add_argument('--level', type=str, default="ExpertPlus", choices=["Easy", "Normal", "Hard", "Expert", "Expert"])
    parser.add_argument('--audio_input_length', type=int, default=6 * 60)
    parser.add_argument('--map_input_length', type=int, default=12 * 60)
    parser.add_argument('--map_target_length', type=int, default=20)
    parser.add_argument('--map_target_shift', type=int, default=12*60)
    parser.add_argument('--map_dimensions', type=int, default=12)
    parser.add_argument('--audio_dimensions', type=int, default=35)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--audio_fr', type=int, default=60)
    parser.add_argument('--map_fr', type=int, default=60)

    # Training options
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)

    # Model options
    parser.add_argument('--model', type=str, default='fact', choices=['fact'])

    parser.add_argument('--cm_hidden_size', type=int, default=800)
    parser.add_argument('--cm_num_layers', type=int, default=10)
    parser.add_argument('--cm_num_heads', type=int, default=8)
    parser.add_argument('--cm_intermediate_size', type=int, default=3072)
    parser.add_argument('--cm_out_size', type=int, default=12*20)

    parser.add_argument('--map_hidden_size', type=int, default=800)
    parser.add_argument('--map_num_layers', type=int, default=2)
    parser.add_argument('--map_num_heads', type=int, default=10)
    parser.add_argument('--map_intermediate_size', type=int, default=3072)

    parser.add_argument('--audio_hidden_size', type=int, default=800)
    parser.add_argument('--audio_num_layers', type=int, default=2)
    parser.add_argument('--audio_num_heads', type=int, default=10)
    parser.add_argument('--audio_intermediate_size', type=int, default=3072)

    # Mode options
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--test_dir', type=str, default="./evaluation/test/")
    parser.add_argument('--weight_path', type=str, default="./checkpoint/model_2.pth")
    parser.add_argument('--step', type=int, default=240)

    args = parser.parse_args()
    return args

weights = [1.0000e+02, 3.3490e+03, 3.4680e+03, 3.3490e+03, 3.1690e+03, 1.8424e+04,
 6.5510e+03, 5.7800e+03, 7.7570e+03, 6.0160e+03, 8.9330e+03, 9.2120e+03,
 8.4220e+03, 7.5580e+03, 8.1880e+03, 7.1900e+03, 1.0528e+04, 1.9653e+04,
 1.0165e+04, 7.3699e+04]

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beat_dataset = BeatSaberDataset(args)
    beat_dataloader = torch.utils.data.DataLoader(beat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    model = FACT(args)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    if args.mode == "train":
        train(args, model.to(device), beat_dataloader, device, criterion, optimizer)
    else:
        evaluation(model, args)
