import soundfile
import librosa
import numpy as np
import pickle
import os
from convert_wavs import convert_audio


AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", 
    "boredom"
}


def get_label(audio_config):
    """
    audio_config = {'mfcc': True, 'chroma': True, 'mel': True}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    Trich xuat dac diem tu tep am thanh `file_name`
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            pass
    except RuntimeError:
        # không định dạng đúng thì chuyển đổi sang 16000Hz và mono dùng ffmpeg
        # lấy tên mặc định
        basename = os.path.basename(file_name)
        dirname  = os.path.dirname(file_name)
        name, ext = os.path.splitext(basename)
        new_basename = f"{name}_0.wav"
        new_filename = os.path.join(dirname, new_basename)
        v = convert_audio(file_name, new_filename)
        if v:
            raise NotImplementedError("Chuyển đổi không đúng, nếu không được hãy tải ffmpeg và cài trên máy tính ở Path.")
    else:
        new_filename = file_name
    with soundfile.SoundFile(new_filename) as sound_file:
        X = sound_file.read(dtype="float64")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
    return result


def get_best_estimators(classification):
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
    Chuyen doi danh sach dac trung (feature) thanh tu dien de hieu tu loai
    `data_extractor.AudioExtractor`
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Đặc trưng truyền vào: {feature} không được chấp nhận.")
        audio_config[feature] = True
    return audio_config
    