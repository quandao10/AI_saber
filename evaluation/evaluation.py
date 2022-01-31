import torch
import torch.nn as nn
from dataset.module.audio_preprocessing import extract_audio_feature
import librosa
import numpy as np
import json
import os

HOP_LENGTH = 512
FPS = 60
SR = FPS * HOP_LENGTH

def preprocessing_song(song_file, bpm):
    data, _ = librosa.load(song_file, sr=SR)
    audio_feature = extract_audio_feature(data, bpm)
    return audio_feature


def preprocessing_map(map_file):
    difficulty_content = json.loads(open(map_file).read())
    notes = difficulty_content["_notes"]
    sorted_notes = sorted(notes, key=lambda x: x["_time"])
    song_duration = sorted_notes[-1]["_time"]
    map_feature = np.zeros([int(song_duration * 60) + 1, 12])
    for note in notes:
        position_id = 4 * note["_lineLayer"] + note["_lineIndex"]
        value_id = 19 if note["_type"] == 3 else 1 + 9 * note["_type"] + note["_cutDirection"]
        timer_sec = int(note["_time"] * 60)
        map_feature[timer_sec, position_id] = value_id
    return map_feature


def evaluation(model, args):
    
    files = [args.test_dir + test for test in os.listdir(args.test_dir)]
    model.load_state_dict(torch.load(args.weight_path))
    
    for file in files:
        song_file = file + "/song.egg"
        map_file = file + "/diff.dat"
        bpm = int(open(file + "/Info.dat", "r").read().strip())

        audio_input = preprocessing_song(song_file, bpm)
        map_input = preprocessing_map(map_file)
        map_input = map_input[:args.map_input_length]

        audio_input = torch.as_tensor(audio_input).float()
        map_input = torch.as_tensor(map_input).float()[:args.map_input_length, :]
        
        # pred_map = model(map_input, audio_input)[0]
        # print(torch.argmax(pred_map, dim=-1)[0:60, :])
        
        with torch.no_grad():
            pred_map = model.infer_auto_regressive(map_input, audio_input, bpm, args)
            print(pred_map)
            map = torch.concat([map_input, pred_map], dim=0)
            map = map.cpu().numpy()
            note = decode_map(map)
            with open(file + "/pred_{}.dat".format(args.level), "w") as f:
                f.write(json.dumps(note)) 
    return 0


def get_position_decoder(value):
    """
    return line_layer, line_index
    """
    return int(value//4), int(value%4)


def get_note_rep_decoder(value):
    """
    return type, cutDirection
    """
    if value == 19:
        return 3, 1
    else:
        return int((value-1)//9), int((value-1)%9)


def decode_map(data):
    non_zero_ids = np.argwhere(data>0)
    notes = []
    for ids in non_zero_ids:
        row, col = ids
        line_layer, line_index = get_position_decoder(col)
        note_type, cut_direction = get_note_rep_decoder(data[row][col])
        if note_type + cut_direction != 0:
            beat_time = np.round(row/60, decimals=3)
            note = {"_time": beat_time, "_lineIndex": line_index, "_lineLayer": line_layer, "_type": note_type, "_cutDirection": cut_direction}
            notes.append(note)
    return {"_notes": notes}


 

