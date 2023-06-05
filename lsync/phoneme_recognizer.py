from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, logging, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import torch
from .config import TARGET_SR, MODELS
from .lyrics_processor import LyricsProcessor
from .util import get_audio_segments

logging.set_verbosity_warning()


class PhonemeRecognizer():
    def __init__(self, lang="en-US") -> None:
        model_id = MODELS[lang]
        if lang.endswith('-base'):
            tokenizer = Wav2Vec2CTCTokenizer(
                "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
            self.processor = Wav2Vec2Processor(
                feature_extractor=feature_extractor, tokenizer=tokenizer)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id)
        self.lp = LyricsProcessor(lang=lang)
        self.lang = lang

    def recognize(self, vocals):
        self.duration_sec = 0
        segs = get_audio_segments(vocals)
        logits = self.__recognize(segs[0])
        for seg in segs[1:]:
            logits_seg = self.__recognize(seg)
            logits = torch.cat((logits, logits_seg), dim=1)

        # normalize the emission with log_softmax() to avoid numerical instability by computing prob in log-domain
        emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()
        pred = self.get_pred(logits)
        transcription = self.get_transcription(pred)

        # get real time
        frame_duration = self.model.config.inputs_to_logits_ratio / TARGET_SR
        return emission, pred, transcription, frame_duration

    def tokenize(self, text_path):
        text = self.lp.process(text_path)
        return self.processor.tokenizer(text).input_ids

    def __recognize(self, vocals):
        vals = self.processor(vocals, return_tensors="pt",
                              padding="longest", sampling_rate=16000).input_values
        self.duration_sec += vals.shape[1] / TARGET_SR
        with torch.no_grad():
            logits = self.model(vals).logits
        return logits.cpu().detach()

    def get_labels(self):
        return [k for k, v in sorted(self.processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]

    def get_pred(self, logits):
        return torch.argmax(logits, dim=-1)

    def get_transcription(self, pred):
        return self.processor.batch_decode(pred)
