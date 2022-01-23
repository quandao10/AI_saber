import argparse
import torch
from dataset.dataset import BeatSaberDataset
from model.fact_model import FACT
from train.train import train
import os
from torchvision.transforms import Compose, ToTensor, Normalize

args_sample = {
    "map_path": "./dataset/data/map_feature/",
    "audio_path": "./dataset/data/audio_feature/",
    "difficulty_path": "./dataset/data/difficulty_list/easy_file.pkl",
    "level": "Easy",
    "audio_input_length": 10 * 60, # 10 seconds
    "map_input_length": 20 * 48, # 20 beats
    "map_target_length": 10 * 48, # 10 beats
    "map_target_shift": 20 * 48, # 20 beats
    "bpm_path": "./dataset/data/audio_tempo.json",
    "map_dimensions": 12,
    "audio_dimensions": 35,
    "batch_size": 32,
    "epochs": 100,
    "lr": 0.001,
    "momentum": 0.9,
    "seed": 1,
    "log_interval": 10,
    "model": "fact",
    "cm_hidden_size": 800,
    "cm_num_layers": 12,
    "cm_num_heads": 12,
    "cm_intermediate_size": 3072,
    "cm_out_size": 12,
    "map_hidden_size": 800,
    "map_num_layers": 2,
    "map_num_heads": 10,
    "map_intermediate_size": 3072,
    "audio_hidden_size": 800,
    "audio_num_layers": 2,
    "audio_num_heads": 10,
    "audio_intermediate_size": 3072,
    "num_classes": 20,
}

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Data
    parser.add_argument('--map_path', type=str, default="./dataset/data/map_feature_60/")
    parser.add_argument('--difficulty_path', type=str, default="./dataset/data/difficulty_list/expert_plus_file.pkl")
    parser.add_argument('--audio_path', type=str, default="./dataset/data/audio_feature/")
    parser.add_argument('--bpm_path', type=str, default="./dataset/data/audio_tempo.json")
    parser.add_argument('--level', type=str, default="ExpertPlus", choices=["Easy", "Normal", "Hard", "Expert", "Expert"])
    parser.add_argument('--audio_input_length', type=int, default=3 * 60)
    parser.add_argument('--map_input_length', type=int, default=4 * 60)
    parser.add_argument('--map_target_length', type=int, default=2 * 60)
    parser.add_argument('--map_target_shift', type=int, default=4 * 60)
    parser.add_argument('--map_dimensions', type=int, default=12)
    parser.add_argument('--audio_dimensions', type=int, default=35)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--audio_fr', type=int, default=60)
    parser.add_argument('--map_fr', type=int, default=60)

    # Training options
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)

    # Model options
    parser.add_argument('--model', type=str, default='fact', choices=['fact'])

    parser.add_argument('--cm_hidden_size', type=int, default=400)
    parser.add_argument('--cm_num_layers', type=int, default=10)
    parser.add_argument('--cm_num_heads', type=int, default=8)
    parser.add_argument('--cm_intermediate_size', type=int, default=2048)
    parser.add_argument('--cm_out_size', type=int, default=12*20)

    parser.add_argument('--map_hidden_size', type=int, default=400)
    parser.add_argument('--map_num_layers', type=int, default=2)
    parser.add_argument('--map_num_heads', type=int, default=10)
    parser.add_argument('--map_intermediate_size', type=int, default=2048)

    parser.add_argument('--audio_hidden_size', type=int, default=400)
    parser.add_argument('--audio_num_layers', type=int, default=2)
    parser.add_argument('--audio_num_heads', type=int, default=10)
    parser.add_argument('--audio_intermediate_size', type=int, default=2048)

    # Mode options
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    beat_dataset = BeatSaberDataset(args)
    beat_dataloader = torch.utils.data.DataLoader(beat_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    model = FACT(args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    if args.mode == "train":
        train(args, model, beat_dataloader, device, criterion, optimizer)
    else:
        pass
