"""
Utility functions for hear-kit
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

def compute_stats(melspec):
        # Compute mean, std
        mean = torch.sum(melspec.mean())
        std = torch.sum(melspec.std())

        mean /= len(melspec)
        std /= len(melspec)

        stats = [mean.item(), std.item()]
        return stats

def frame_audio(
    audio: Tensor, frame_size: int, hop_size: float, sample_rate: int
) -> Tuple[Tensor, Tensor]:
    """
    Slices input audio into frames that are centered and occur every
    sample_rate * hop_size samples. We round to the nearest sample.
    Args:
        audio: input audio, expects a 2d Tensor of shape:
            (n_sounds, num_samples)
        frame_size: the number of samples each resulting frame should be
        hop_size: hop size between frames, in milliseconds
        sample_rate: sampling rate of the input audio
    Returns:
        - A Tensor of shape (n_sounds, num_frames, frame_size)
        - A Tensor of timestamps corresponding to the frame centers with shape:
            (n_sounds, num_frames).
    """

    # Zero pad the beginning and the end of the incoming audio with half a frame number
    # of samples. This centers the audio in the middle of each frame with respect to
    # the timestamps.
    audio = F.pad(audio, (frame_size // 2, frame_size - frame_size // 2))
    num_padded_samples = audio.shape[1]

    frame_step = hop_size / 1000.0 * sample_rate
    frame_number = 0
    frames = []
    timestamps = []
    frame_start = 0
    frame_end = frame_size
    while True:
        frames.append(audio[:, frame_start:frame_end])
        timestamps.append(frame_number * frame_step / sample_rate * 1000.0)

        # Increment the frame_number and break the loop if the next frame end
        # will extend past the end of the padded audio samples
        frame_number += 1
        frame_start = int(round(frame_number * frame_step))
        frame_end = frame_start + frame_size

        if not frame_end <= num_padded_samples:
            break

    # Expand out the timestamps to have shape (n_sounds, num_frames)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
    timestamps_tensor = timestamps_tensor.expand(audio.shape[0], -1)

    return torch.stack(frames, dim=1), timestamps_tensor