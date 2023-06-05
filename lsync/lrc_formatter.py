from dataclasses import dataclass
from typing import List

def seconds_to_lrc(seconds, is_word = True):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    hundredths = int((seconds % 1) * 100)
    seconds = int(seconds)
    formated = f"{minutes:02d}:{seconds:02d}.{hundredths:02d}"
    return f"<{formated}>" if is_word else f"[{formated}]"


@dataclass
class Word:
    label: str
    start: float
    end: float

    def __repr__(self):
        return f"{seconds_to_lrc(self.start)} {self.label}"

class LrcFormatter():
    @staticmethod
    def words2lrc(words: List[Word], original_lyrics: str, lang="en-US"):
        lrc = ""
        counter = 0
        word_end = None
        for line in original_lyrics.splitlines():
            if line == '': continue
            if word_end:
                lrc += f"\n{seconds_to_lrc(word_end, False)}"
            else:
                lrc += "[00:00.00]"
            if lang == "en-US":
                splitted_words = line.split(' ')
            elif lang == "zh-CN":
                splitted_words = line

            for original_word in splitted_words:
                if original_word == '': continue
                word = words[counter]
                word.label = original_word
                lrc += f" {word}"
                word_end = word.end
                counter+=1
        return lrc
