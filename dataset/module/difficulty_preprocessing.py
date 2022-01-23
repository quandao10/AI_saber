import json
import os
import zipfile
import numpy as np


def valid_note_check(note):
    return note["_type"] in range(0, 4) and note["_cutDirection"] in range(0, 9) and note["_lineIndex"] in range(0, 4) \
           and note["_lineLayer"] in range(0, 3) and note["_type"] != 2 and note["_time"] >= 0


def sort_invalid_map(data_dir, output_dir):
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.zip')]
    with open(output_dir + 'audio_ignore_list.txt', 'r') as f:
        ignore_list = f.read().splitlines()
    file_list = [file for file in file_list if file not in ignore_list]
    map_ignore_list = set()
    for i, file in enumerate(file_list):
        valid = True
        with zipfile.ZipFile(data_dir + file, 'r') as zip_ref:
            try:
                for name_file in zip_ref.namelist():
                    if name_file[:-4].lower() == "info":
                        infor = json.loads(zip_ref.read(name_file).decode("utf-8"))
                        break
                difficulty_list = infor["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
                for difficulty in difficulty_list:
                    difficulty_file = difficulty["_beatmapFilename"]
                    difficulty_content = json.loads(zip_ref.read(difficulty_file).decode('utf-8'))
                    notes = difficulty_content["_notes"]
                    if len(notes) < 100:
                        valid = False
                    for note in notes:
                        if not valid_note_check(note):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    map_ignore_list.add(file)
            except:
                map_ignore_list.add(file)
        if i % 100 == 0:
            print("{}/{} processed".format(i, len(file_list)))
    with open(output_dir + 'map_ignore_list.txt', 'w') as f:
        f.write('\n'.join(map_ignore_list))
        print("{} maps ignored".format(len(map_ignore_list)))


# sort_invalid_map('../../../beat_saber/dataset/', '../data/')


def convert_bpm_sec(bpm, beat_time):
    return 60 / bpm * beat_time


def preprocess_map_note(data_dir, ignore_audio_files, ignore_map_files, short_song_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(ignore_audio_files, 'r') as f:
        ignore_audio_list = f.read().splitlines()

    with open(ignore_map_files, 'r') as f:
        ignore_map_list = f.read().splitlines()
        
    with open(short_song_files, 'r') as f:
        short_audio_list = f.read().splitlines()
        
    short_audio_list = [file[:-4] + '.zip' for file in short_audio_list]

    ignore_list = ignore_audio_list + ignore_map_list + short_audio_list
    
    ignore_list = list(set(ignore_list))
    
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.zip') and file not in ignore_list]

    for i, file in enumerate(file_list):
        with zipfile.ZipFile(data_dir + file, 'r') as zip_ref:
            for name_file in zip_ref.namelist():
                if name_file[:-4].lower() == "info":
                    infor = json.loads(zip_ref.read(name_file).decode("utf-8"))
                    break
            difficulty_list = infor["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
            for difficulty in difficulty_list:
                difficulty_file = difficulty["_beatmapFilename"]
                difficulty_content = json.loads(zip_ref.read(difficulty_file).decode('utf-8'))
                notes = difficulty_content["_notes"]
                sorted_notes = sorted(notes, key=lambda x: x["_time"])
                song_duration = sorted_notes[-1]["_time"]
                map_feature = np.zeros([int(song_duration * 60) + 1, 12])
                for note in notes:
                    position_id = 4 * note["_lineLayer"] + note["_lineIndex"]
                    value_id = 19 if note["_type"] == 3 else 1 + 9 * note["_type"] + note["_cutDirection"]
                    timer_sec = int(note["_time"] * 60)
                    map_feature[timer_sec, position_id] = value_id
                np.save(output_dir + file[:-4] + "_" + difficulty["_difficulty"] + ".npy", map_feature)
        if i % 100 == 0:
            print("{}/{} processed map".format(i, len(file_list)))
    print("Finish preprocessing map")


def get_position_decoder(value):
    """
    return line_layer, line_index
    """
    return int(value // 4), int(value % 4)


def get_note_rep_decoder(value):
    """
    return type, cutDirection
    """
    if value == 19:
        return 3, 1
    else:
        return int((value - 1) // 9), int((value - 1) % 9)


def decode_map(np_file):
    data = np.load(np_file)
    non_zero_ids = np.argwhere(data > 0)
    notes = []
    for ids in non_zero_ids:
        row, col = ids
        line_layer, line_index = get_position_decoder(col)
        note_type, cut_direction = get_note_rep_decoder(data[row][col])
        beat_time = np.round(row / 60, decimals=3)
        note = {"_time": beat_time, "_lineIndex": line_index, "_lineLayer": line_layer, "_type": note_type,
                "_cutDirection": cut_direction}
        notes.append(str(note))
    with open(np_file[:-4] + "_decoded.dat", "w") as file:
        file.write("\n".join(notes))
    return 0


# preprocess_map_note("../beat_saber/dataset/",
#                     "dataset/data/audio_ignore_list.txt",
#                     "dataset/data/map_ignore_list.txt",
#                     "dataset/data/short_song.txt",
#                     "dataset/data/map_feature_60/")
