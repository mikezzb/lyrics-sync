from .voice_extractor import VoiceExtractor
from .phoneme_recognizer import PhonemeRecognizer
from .alignment import Aligner
from .lyrics_processor import LyricsProcessor
from .util import read_text, save_lrc, save_words, save_audio
from .config import TARGET_SR
from .lrc_formatter import LrcFormatter
import os

class LyricsSync():
    def __init__(self, lang="en-US", blank_id=0) -> None:
        self.ve = VoiceExtractor()
        self.phone_rec = PhonemeRecognizer(lang=lang)
        self.lp = LyricsProcessor(lang=lang)
        self.blank_id = blank_id

    def sync(self, audio_fn: str, text_fn: str, save=True):
        audio_name = os.path.splitext(os.path.basename(audio_fn))[0]
        vocals = self.ve.extract_voice(audio_fn)
        if save:
            save_audio(vocals, audio_name, TARGET_SR)
        emission, _, _, frame_duration = self.phone_rec.recognize(
            vocals)

        # Preprocessing & tokenization
        tokens = self.phone_rec.tokenize(text_fn)
        path = Aligner.align(emission, tokens, blank_id=self.blank_id)
        processed_lyrics = self.lp.process(text_fn)
        words = self.lp.get_words_from_path(
            processed_lyrics, path, frame_duration)

        # lrc
        original_lyrics = read_text(text_fn)
        lrc = LrcFormatter.words2lrc(words, original_lyrics)
        if save:
            save_words(words, audio_name)
            save_lrc(lrc, audio_name)

        return words, lrc
