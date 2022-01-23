import json
import os
import zipfile
import pickle


def get_list_of_difficulty_files(file_dir, audio_ignore_dir, map_ignore_dir, short_file_dir):
    zip_list = [file for file in os.listdir(file_dir) if file.endswith(".zip")]
    easy_file = {}
    normal_file = {}
    hard_file = {}
    expert_file = {}
    expert_plus_file = {}
    with open(audio_ignore_dir, "r") as f:
        audio_ignore = f.read().splitlines()
    with open(map_ignore_dir, "r") as f:
        map_ignore = f.read().splitlines()
    with open(short_file_dir, "r") as f:
        short_ignore = f.read().splitlines()
    short_ignore = [file[:-4] + ".zip" for file in short_ignore]
    ignore_file = audio_ignore + map_ignore + short_ignore
    ignore_file = set(ignore_file)
    zip_list = [file for file in zip_list if file not in ignore_file]
    for i, zip_name in enumerate(zip_list):
        with zipfile.ZipFile(os.path.join(file_dir, zip_name), 'r') as zip_ref:
            for name_file in zip_ref.namelist():
                if name_file[:-4].lower() == "info":
                    infor = json.loads(zip_ref.read(name_file).decode("utf-8"))
                    break
            try:
                difficulty_list = infor["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"]
            except KeyError as e:
                print(zip_name, zip_ref.namelist())

            for difficulty in difficulty_list:
                if difficulty["_difficulty"] == "Easy":
                    easy_file[zip_name] = difficulty["_beatmapFilename"]
                elif difficulty["_difficulty"] == "Normal":
                    normal_file[zip_name] = difficulty["_beatmapFilename"]
                elif difficulty["_difficulty"] == "Hard":
                    hard_file[zip_name] = difficulty["_beatmapFilename"]
                elif difficulty["_difficulty"] == "Expert":
                    expert_file[zip_name] = difficulty["_beatmapFilename"]
                else:
                    expert_plus_file[zip_name] = difficulty["_beatmapFilename"]
        if i % 100 == 0:
            print("{}/{} done writing".format(i, len(zip_list)))
    pickle.dump(easy_file, open("dataset/data/difficulty_list/easy_file.pkl", "wb"))
    pickle.dump(normal_file, open("dataset/data/difficulty_list/normal_file.pkl", "wb"))
    pickle.dump(hard_file, open("dataset/data/difficulty_list/hard_file.pkl", "wb"))
    pickle.dump(expert_file, open("dataset/data/difficulty_list/expert_file.pkl", "wb"))
    pickle.dump(expert_plus_file, open("dataset/data/difficulty_list/expert_plus_file.pkl", "wb"))
    return 0


# get_list_of_difficulty_files("../beat_saber/dataset", "dataset/data/audio_ignore_list.txt",
#                              "dataset/data/map_ignore_list.txt",  "dataset/data/short_song.txt")


def load_pkl_file(file_name):
    with open(file_name, "rb") as f:
        pickle_file = pickle.load(f)
        return pickle_file
    
    
map_level = {
    "Easy": "easy",
    "Normal": "normal",
    "Hard": "hard",
    "Expert": "expert",
    "ExpertPlus": "expert_plus"
}
    

def verify_difficulty_file(difficulty_file_dir, map_dir):
    print("Verifying difficulty file")
    for level in ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]:
        print("Verifying {}".format(level))
        file_list = load_pkl_file(difficulty_file_dir+"{}_file.pkl".format(map_level[level]))
        for zip_name, beatmap_name in file_list.items():
            file_name = zip_name[:-4] + "_" + level + ".npy"
            existed = os.path.exists(map_dir + file_name)
            if not existed:
                print("{} not existed".format(file_name))
        print("Verifying {} done".format(level))
        
        

# verify_difficulty_file("dataset/data/difficulty_list/", "dataset/data/map_feature_60/")


# data = load_pkl_file("dataset/data/difficulty_list/expert_plus_file.pkl")
# del data["cceed0e357601240069a2cdd529505d3c8bac6b5.zip"]
# del data["195fad64589a3b4828979b7d617815c55e887ecb.zip"]
# with open("dataset/data/difficulty_list/expert_plus_file.pkl", "wb") as f:
#     pickle.dump(data, f)