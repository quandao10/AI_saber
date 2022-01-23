import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import *
from argparse import Namespace


class FACT(nn.Module):
    def __init__(self, args):
        super(FACT, self).__init__()
        self.args = args
        self.cross_modal_layer = CrossModalLayer(hidden_size=self.args.cm_hidden_size,
                                                 num_attention_heads=self.args.cm_num_heads,
                                                 intermediate_size=self.args.cm_intermediate_size,
                                                 out_dim=self.args.cm_out_size,
                                                 num_hidden_layers=self.args.cm_num_layers)
        self.map_transformer = TransformerEncoder(hidden_size=self.args.map_hidden_size,
                                                  num_hidden_layers=self.args.map_num_layers,
                                                  num_attention_heads=self.args.map_num_heads,
                                                  intermediate_size=self.args.map_intermediate_size)
        self.map_pos_embedding = PositionalEncoding(dim=self.args.map_hidden_size,
                                                    seq_length=self.args.map_input_length)
        self.map_linear_embedding = LinearEmbedding(in_dim=self.args.map_dimensions,  # need investigate
                                                    out_dim=self.args.map_hidden_size)
        self.audio_transformer = TransformerEncoder(hidden_size=self.args.audio_hidden_size,
                                                    num_hidden_layers=self.args.audio_num_layers,
                                                    num_attention_heads=self.args.audio_num_heads,
                                                    intermediate_size=self.args.audio_intermediate_size)
        self.audio_pos_embedding = PositionalEncoding(dim=self.args.audio_hidden_size,
                                                      seq_length=self.args.audio_input_length)
        self.audio_linear_embedding = LinearEmbedding(in_dim=self.args.audio_dimensions,
                                                      out_dim=self.args.audio_hidden_size)

    def forward(self, map_input, audio_input):
        map_feature = self.map_linear_embedding(map_input)
        map_feature = self.map_pos_embedding(map_feature)
        # print(map_feature.size())
        map_feature = self.map_transformer(map_feature)

        audio_feature = self.audio_linear_embedding(audio_input)
        audio_feature = self.audio_pos_embedding(audio_feature)
        audio_feature = self.audio_transformer(audio_feature)
        # print(audio_feature.size())

        output = self.cross_modal_layer(map_feature, audio_feature)
        return output

    def infer_auto_regressive(self, map_input, audio_input, step=2400):
        outputs = []
        audio_seq_len = self.args.audio_input_length
        for i in range(step):
            audio_input = audio_input[:, i:i + audio_seq_len]
            if audio_input.size(1) < audio_seq_len:
                break
            output = self.forward(map_input, audio_input)
            output = output[:, 0:48, :]  # only keep the first 48 frames (first beat)
            outputs.append(output)
            map_input = torch.cat([map_input[:, 48:, :], output], dim=1)
        return torch.cat(outputs, dim=1)

    def compute_loss(self, pred, target):
        _, target_len_seq, _ = target.size()
        diff = pred - target[:, :target_len_seq, :]
        l2_loss = torch.norm(diff, dim=2)
        return l2_loss.mean()


args_sample = {
    "map_path": "./data/map_feature/",
    "audio_path": "./data/audio_feature/",
    "difficulty_path": "./data/difficulty_list/easy_file.pkl",
    "level": "Easy",
    "audio_input_length": 10 * 60,  # 10 seconds
    "map_input_length": 20 * 48,  # 20 beats
    "map_target_length": 10 * 48,  # 10 beats
    "map_target_shift": 20 * 48,  # 20 beats
    "bpm_path": "./data/audio_tempo.json",
    "map_dimensions": 12,
    "audio_dimensions": 35,
    "batch_size": 16,
    "epochs": 100,
    "lr": 0.001,
    "momentum": 0.9,
    "seed": 1,
    "log_interval": 10,
    "model": "fact",
    "cm_hidden_size": 800,
    "cm_num_layers": 2,
    "cm_num_heads": 10,
    "cm_intermediate_size": 1024,
    "cm_out_size": 12 * 19,
    "map_hidden_size": 800,
    "map_num_layers": 2,
    "map_num_heads": 10,
    "map_intermediate_size": 1024,
    "audio_hidden_size": 800,
    "audio_num_layers": 2,
    "audio_num_heads": 10,
    "audio_intermediate_size": 3072,
    "num_classes": 20,
}
# arg_ns = Namespace(**args_sample)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FACT(arg_ns).to(device)
# audio_input = torch.randn(args_sample["batch_size"], args_sample["audio_input_length"], args_sample["audio_dimensions"]).to(device)
# map_input = torch.randint(0, 19, (args_sample["batch_size"], args_sample["map_input_length"], args_sample["map_dimensions"]), dtype=torch.float32).to(device)
#
# print(audio_input.size(), map_input.size())
# output = model(map_input, audio_input)
# print(output.size())
