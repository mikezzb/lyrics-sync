from dataclasses import dataclass
from .util import read_text
from .lrc_formatter import Word

SEPARATOR = '|'


@dataclass
class Segment:
    label: str
    start: int
    end: int


class LyricsProcessor():
    def __init__(self, lang: str = "en-US") -> None:
        self.lang = lang

    def process(self, text_path):
        text = read_text(text_path)
        if self.lang == "en-US":
            return self.__process_en(text)
        elif self.lang == 'zh-CN':
            return self.__process_cn(text)
        elif self.lang.startswith("en") and self.lang.endswith("-base"):
            return self.__process_en(text, is_upper=False)

    def __process_en(self, text: str, is_upper=True):
        # preprocessing
        if is_upper:
            text = text.upper()
        else:
            text = text.lower()
        text = text.replace(' ', '|')
        text = text.replace('\n', '|')
        text = text.replace('_', '\'')
        text = text.replace('’', '\'')
        return text

    def __process_cn(self, text):
        # preprocessing
        return text

    def get_words_from_path(self, text, path, frame_duration):
        # Skip repeating char
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            segments.append(
                Segment(
                    text[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1
                )
            )
            i1 = i2
        if self.lang.startswith("en"):
            return self.__merge_en(segments, frame_duration)
        elif self.lang == 'zh-CN':
            return [Word(s.label, s.start * frame_duration, s.end * frame_duration) for s in segments]

    def __merge_en(self, segments, frame_duration, separator=SEPARATOR):
        # Merge chars to word
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    words.append(Word(
                        word, segments[i1].start * frame_duration, segments[i2 - 1].end * frame_duration))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words
