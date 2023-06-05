import torch
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.separate import load_track
import librosa
from .config import TARGET_SR, ORIGINAL_SR


class VoiceExtractor():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.__load_model()
        self.vocals_source_idx = self.model.sources.index("vocals")
        self.sr = self.model.samplerate

    def extract_voice(self, audio_fn, post_process=True):
        audio = self.__load_audio(audio_fn)
        vocals = self.__extract_voice(audio)
        return vocals if not post_process else self.post_process(vocals)

    def post_process(self, audio, sr=ORIGINAL_SR, target_sr=TARGET_SR):
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio

    def __extract_voice(self, audio: np.array, shifts: int = 1) -> np.array:
        """Extract mono vocals from the audio"""
        ref = audio.mean(0)
        audio = (audio - ref.mean()) / ref.std()
        with torch.no_grad():
            sources = apply_model(
                self.model, audio[None], device=self.device, shifts=shifts, split=True, overlap=0.25, progress=False)
        vocals = sources[0][self.vocals_source_idx]
        return vocals.cpu().numpy()[0, ...]

    def __load_model(self):
        model = get_model(name="htdemucs", repo=None)
        model.to(self.device)
        model.eval()
        return model

    def __load_audio(self, audio_fn: str) -> np.array:
        return load_track(audio_fn, 2, self.sr)
