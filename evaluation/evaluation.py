import torch
import torch.nn as nn





def evaluation(model, weight_path, song_file, map_pad_file):
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    