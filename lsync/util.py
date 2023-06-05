import librosa
import soundfile as sf
from .config import ORIGINAL_SR, TARGET_SR
from lsync.lrc_formatter import Word
from typing import List
import dataclasses
import pandas as pd
import numpy as np

window_size = int(TARGET_SR * 15)
hop_length = window_size


def get_audio_segments(audio):
    """Split audio to segments"""
    return librosa.util.frame(audio, frame_length=window_size, hop_length=hop_length, axis=0)


def get_audio_segments_by_onsets(audio):
    onset_times = librosa.onset.onset_detect(
        y=audio, sr=TARGET_SR, backtrack=True)
    onset_boundaries = np.concatenate([onset_times, [len(audio)]])
    segments = []
    start_onset = 0
    for onset in onset_boundaries:
        segments.append(audio[start_onset:onset])
    return segments


def read_text(text_path):
    with open(text_path, 'r') as file:
        data = file.read()
        return data


def save_audio(audio, name, sr=ORIGINAL_SR):
    sf.write(f'output/vocals/{name}.wav', audio, sr)


def save_audio(audio, name, sr=ORIGINAL_SR, out_path="output/vocals"):
    sf.write(f'{out_path}/{name}.wav', audio, sr)


def save_audio_file(audio, path, sr=ORIGINAL_SR):
    sf.write(f'{path}.wav', audio, sr)


def save_lrc(lrc: str, name: str):
    with open(f'output/lrc/{name}.lrc', 'w+') as fp:
        fp.write(lrc)


def save_words(words: List[Word], name: str):
    df = pd.DataFrame([dataclasses.asdict(w) for w in words])
    df.to_csv(f"output/words/{name}.csv", index=False)
