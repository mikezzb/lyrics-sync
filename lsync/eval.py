import numpy as np
from typing import List
from lsync.lrc_formatter import Word
import dataclasses
import pandas as pd
from tqdm import tqdm
import os
from lsync import LyricsSync


def get_eval_df(dataset='jamendo'):
    df = pd.DataFrame(columns=['name', 'audio', 'lyrics', 'annotation'])
    if dataset == 'jamendo':
        base = "dataset/jamendolyrics"
        songs = os.listdir(f'{base}/mp3')
        for song in songs:
            name, _ = os.path.splitext(song)
            df = df.append({
                'name': name,
                'audio': f"{base}/mp3/{name}.mp3",
                'lyrics': f"{base}/lyrics/{name}.raw.txt",
                'annotation': f"{base}/annotations/words/{name}.csv",
            }, ignore_index=True)
    return df


class Eval:
    @staticmethod
    def eval_all(eval_df: pd.DataFrame, lsync: LyricsSync):
        record = pd.DataFrame(
            columns=['name', 'average_abs_err', 'percentage_of_correct_segments'])
        for idx, row in tqdm(eval_df.iterrows()):
            try:
                words, lrc = lsync.sync(row['audio'], row['lyrics'])
                abs_err, perc_corr_segs = Eval.evaluate(
                    words, row['annotation'])
                record = record.append({
                    'name': row['name'],
                    'average_abs_err': abs_err,
                    'percentage_of_correct_segments': perc_corr_segs
                }, ignore_index=True)

            except Exception as e:
                # print(traceback.format_exc())
                continue
        record.to_csv("eval.csv", index=False)
        return record

    @staticmethod
    def evaluate(words: List[Word], gt_path: str):
        pred_df = pd.DataFrame([dataclasses.asdict(w) for w in words])
        truth_df = Eval.load_ground_truth_jamendo(gt_path)
        abs_err = Eval.average_abs_err(
            truth_df['start'].values, pred_df['start'].values)
        perc_corr_segs = Eval.percentage_of_correct_segments(
            truth_df['start'].values, pred_df['start'].values)
        return abs_err, perc_corr_segs

    @staticmethod
    def average_abs_err(ref, pred):
        deviations = np.abs(ref - pred)
        return np.mean(deviations)

    @staticmethod
    def percentage_of_correct_segments(ref, pred, window=0.3):
        deviations = np.abs(ref - pred)
        return np.mean(deviations <= window)

    @staticmethod
    def load_ground_truth_jamendo(gt_path: str):
        df = pd.read_csv(gt_path, skiprows=1, names=["start", "line_end"])
        return df

    @staticmethod
    def load_ground_truth_hensen(gt_path: str):
        data = np.genfromtxt(gt_path, delimiter='\t', dtype=[
                             ('start', float), ('end', float), ('label', 'U10')])
        filtered = data[data['label'] != 'PAU']
        df = pd.DataFrame(filtered)
        return df
