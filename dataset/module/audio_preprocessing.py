import os
import librosa
import numpy as np
import zipfile
import json
import io
import multiprocessing
from functools import partial

HOP_LENGTH = 512
FPS = 60
SR = FPS * HOP_LENGTH


def extract_audio_feature(data, bpm):
    envelop = librosa.onset.onset_strength(data, sr=SR)
    mfcc = librosa.feature.mfcc(data, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12).T

    peak_idxs = librosa.onset.onset_detect(onset_envelope=envelop, sr=SR, hop_length=HOP_LENGTH)
    peak_onehot = np.zeros_like(envelop, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelop,
                                               sr=SR,
                                               hop_length=HOP_LENGTH,
                                               start_bpm=bpm,
                                               tightness=100.0)

    beat_onehot = np.zeros_like(envelop, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    audio_feature = np.concatenate((envelop[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]),
                                   axis=-1)

    return audio_feature


def audio_preprocess_supporter(file, data_dir, output_path):
    try:
        with zipfile.ZipFile(data_dir + file, 'r') as zip_ref:
            for name_file in zip_ref.namelist():
                if name_file[:-4].lower() == "info":
                    infor = json.loads(zip_ref.read(name_file).decode("utf-8"))
                    break
            with zip_ref.open(infor["_songFilename"]) as myfile:
                data, sr = librosa.load(io.BytesIO(myfile.read()), sr=SR)
                audio_feature = extract_audio_feature(data, infor["_beatsPerMinute"])
                np.save(output_path + file[:-4] + ".npy", audio_feature)
    except Exception as e:
        print(e)
        with open(output_path + "ignore_list.txt", "a") as f:
            f.write(file + "\n")


def audio_preprocess(data_dir, output_dir):
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.zip')]
    output_path = output_dir + "audio_feature/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pool = multiprocessing.Pool(processes=200)
    preprocess_supporter = partial(audio_preprocess_supporter, data_dir=data_dir, output_path=output_path)
    for i, _ in enumerate(pool.imap_unordered(preprocess_supporter, file_list)):
        if i % 100 == 0:
            print("{}/{} extracted audio feature".format(i, len(file_list)))


def extract_audio_tempo(data_dir, output_dir):
    tempo_dict = {}
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.zip')]
    for i, file in enumerate(file_list):
        with zipfile.ZipFile(data_dir + file, 'r') as zip_ref:
            for name_file in zip_ref.namelist():
                if name_file[:-4].lower() == "info":
                    infor = json.loads(zip_ref.read(name_file).decode("utf-8"))
                    break
            tempo = infor["_beatsPerMinute"]
        tempo_dict[file[:-4]] = tempo
        if i % 100 == 0:
            print("{}/{} extracted audio tempo".format(i, len(file_list)))
    json.dump(tempo_dict, open(output_dir + "audio_tempo.json", "w"))
    return 0


# extract_audio_tempo("../../../beat_saber/dataset/", "../data/")


def get_short_song(data_dir, output_dir):
    short_song = []
    file_list = [file for file in os.listdir(data_dir) if file.endswith('.npy')]
    for i, file in enumerate(file_list):
        song_duration = np.load(data_dir + file).shape[0]/60
        if song_duration < 30:
            short_song.append(file)
        if i % 100 == 0:
            print("{}/{} short song".format(i, len(file_list)))
    with open(output_dir+"short_song.txt", "w") as f:
        for song in short_song:
            f.write(song + "\n")
            
# get_short_song("data/audio_feature/", "data/")




# audio_preprocess("../../../beat_saber/dataset/", "../../../beat_saber/output/")
