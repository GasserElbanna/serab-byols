"""
Hear Competition submission script following the 
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api
guidelines
"""

from collections import OrderedDict
import math
from typing import Tuple

import librosa
import torch
import torch.fft
from torch import Tensor
from torchaudio.transforms import MelSpectrogram
from byol_a.augmentations import PrecomputedNorm
from byol_a.models.audio_ntt import AudioNTT2020

from utils import *


# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


def load_model(model_file_path: str = "") -> torch.nn.Module:
    """Load pre-trained DL models.

    Parameters
    ----------


    Returns
    -------
    torch.nn.Module object or a tensorflow "trackable" object
        Model loaded with pre-training weights

    Raises
    ------
    ValueError
        If the model configuration is erroneous
        N.B. A PyTorch model should be loaded with pre-trained weights
    """
    # Load pretrained weights.
    # if model_name == 'byols':
    model = AudioNTT2020(n_mels=64, d=2048)
    
    state_dict = torch.load(model_file_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    return model

def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size: float = TIMESTAMP_HOP_SIZE,
) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps. Both
    the embeddings and corresponding timestamps (in milliseconds) are returned.
    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1].
        model: Loaded model.
        hop_size: Hop size in milliseconds.
            NOTE: Not required by the HEAR API. We add this optional parameter
            to improve the efficiency of scene embedding.
    Returns:
        - Tensor: embeddings, A float32 Tensor with shape (n_sounds, n_timestamps,
            model.timestamp_embedding_size).
        - Tensor: timestamps, Centered timestamps in milliseconds corresponding
            to each embedding in the output. Shape: (n_sounds, n_timestamps).
    """

    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    # Make sure the correct model type was passed in
    if not isinstance(model, AudioNTT2020):
        raise ValueError(
            f"Model must be an instance of {AudioNTT2020.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=1024,
        hop_size=hop_size,
        sample_rate=AudioNTT2020.sample_rate,
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = frames.flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    # Put the model into eval mode, and not computing gradients while in inference.
    # Iterate over all batches and accumulate the embeddings for each frame.
    # Disable parameter tuning
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    """
    This function returns a single embedding for each audio clip. In this baseline
    implementation we simply summarize the temporal embeddings from
    get_timestamp_embeddings() using torch.mean().
    Args:
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in
            a batch will be padded/trimmed to the same length.
        model: Loaded model.
    Returns:
        - embeddings, A float32 Tensor with shape
            (n_sounds, model.scene_embedding_size).
    """
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings

if __name__ == '__main__':
    file_path = '/home/gelbanna/hdd/serab_byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth'
    model = load_model(file_path)
    print(model.timestamp_embedding_size)