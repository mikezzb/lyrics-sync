import torch
from dataclasses import dataclass


@dataclass
class Point:
    token_index: int
    time_index: int
    prob: float


class Aligner():
    @staticmethod
    def align(emission, tokens, blank_id=0):
        trellis = Aligner.get_trellis(emission, tokens, blank_id=blank_id)
        path = Aligner.backtrack(trellis, emission, tokens, blank_id=blank_id)
        return path

    @staticmethod
    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    @staticmethod
    def backtrack(trellis, emission, tokens, blank_id=0):
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
            prob = emission[t - 1, tokens[j - 1]
                            if changed > stayed else 0].exp().item()
            path.append(Point(j - 1, t - 1, prob))
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise Exception("Failed")
        return path[::-1]
