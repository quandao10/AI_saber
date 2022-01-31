import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_model import *
import numpy as np


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

    def infer_auto_regressive(self, map_feature, audio_feature, bpm, args):
        outputs = []
        audio_seq_len = args.audio_input_length
        map_input = map_feature
        bps = int(np.floor(bpm/60))
        if bps == 0:
            bps = 1
        # bf = bps * args.map_fr
        for i in range(args.step):
            # step = i * args.audio_fr
            audio_input = audio_feature[i:i+audio_seq_len, :]
            if audio_input.size(0) < audio_seq_len:
                # print(audio_input.size())
                break
            output = self.forward(map_input, audio_input)[0]
            output = torch.argmax(output, dim=-1)
            output = output[0:bps, :]  # only keep the first frames (first beat)
            outputs.append(output)
            map_input = torch.cat([map_feature[bps:, :], output], dim=0)
        return torch.cat(outputs, dim=0)

    def compute_loss(self, pred, target):
        _, target_len_seq, _ = target.size()
        diff = pred - target[:, :target_len_seq, :]
        l2_loss = torch.norm(diff, dim=2)
        return l2_loss.mean()
