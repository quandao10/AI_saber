import os
from torch.utils.data import Dataset
import numpy as np
import json
from dataset.module.file_utils import load_pkl_file
from argparse import Namespace


class BeatSaberDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.map_path = args.map_path
        self.audio_path = args.audio_path
        self.level = args.level
        self.audio_fr = args.audio_fr
        self.map_fr = args.map_fr
        self.audio_input_length = args.audio_input_length
        self.map_input_length = args.map_input_length
        self.map_target_length = args.map_target_length
        self.map_target_shift = args.map_target_shift
        self.difficulty_dict = load_pkl_file(args.difficulty_path)
        self.difficulty_list = list(self.difficulty_dict.keys())
        self.bpm_dict = json.loads(open(args.bpm_path).read())

    def __getitem__(self, idx):
        assert idx < len(self), "index range error"
        zip_name = self.difficulty_list[idx]
        # print(zip_name)
        audio_feature = np.load(os.path.join(self.audio_path, zip_name[:-4] + ".npy"))
        map_feature = np.load(os.path.join(self.map_path, zip_name[:-4] + "_" + self.level + ".npy"))
        bpm = self.bpm_dict[zip_name[:-4]]

        audio_length = audio_feature.shape[0]
        map_length = map_feature.shape[0]
        # print("audio_length: ", audio_length, ". map_length: ", map_length)
        audio_sec_length = audio_length / self.audio_fr
        map_beat_length = map_length / self.map_fr
        map_sec_length = map_beat_length * 60/bpm
        # print("audio_sec_length: ", audio_sec_length, ". map_sec_length: ", map_sec_length)

        truncated_sec_length = min(audio_sec_length, map_sec_length)
        truncated_beat_length = truncated_sec_length * bpm / 60
        # print("truncated_sec_length: ", truncated_sec_length, ". truncated_beat_length: ", truncated_beat_length)

        window_size = max(self.map_input_length, self.map_target_shift + self.map_target_length)
        window_size = max(window_size, self.map_fr * (self.audio_input_length/self.audio_fr) * bpm/60)
        # beat_max = truncated_beat_length * self.map_fr - window_size
        # print("window_size: ", window_size)
        start_beat = np.random.uniform(0, truncated_beat_length * self.map_fr - window_size)
        if start_beat < 0:
            start_beat = 0
        # print("start_beat: ", start_beat)
        start_frame = round(((start_beat/self.map_fr)/bpm) * 60 * self.audio_fr)
        
        start_beat = int(start_beat)

        map_input = map_feature[start_beat:start_beat + self.map_input_length, :]
        map_target = map_feature[
                     start_beat + self.map_target_shift:start_beat + self.map_target_shift + self.map_target_length, :]
        audio_input = audio_feature[start_frame:start_frame + self.audio_input_length, :]

        return map_input, map_target, audio_input

    def __len__(self):
        return len(self.difficulty_list)


args_sample = {
    "map_path": "dataset/data/map_feature_60/",
    "audio_path": "dataset/data/audio_feature/",
    "difficulty_path": "dataset/data/difficulty_list/easy_file.pkl",
    "level": "Easy",
    "audio_input_length": 5 * 60, # 10 seconds
    "map_input_length": 10 * 60, # 20 beats
    "map_target_length": 5 * 60, # 10 beats
    "map_target_shift": 10 * 60, # 20 beats
    "bpm_path": "dataset/data/audio_tempo.json",
    "audio_fr": 60,
    "map_fr": 60
}

# dataset = BeatSaberDataset(Namespace(**args_sample))
# for j in range(20):
#     for i in range(len(dataset)):
#         map_input, map_target, audio_input = dataset[i]
#         if map_input.shape[0] != args_sample["map_input_length"] or \
#                 map_target.shape[0] != args_sample["map_target_length"] or \
#                 audio_input.shape[0] != args_sample["audio_input_length"]:
#             print(i, map_input.shape, map_target.shape, audio_input.shape)

# map_input, map_target, audio_input = dataset[1589]






