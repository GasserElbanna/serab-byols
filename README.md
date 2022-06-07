# BYOL for Speech (BYOL-S)

A speech representation model based on [BYOL-A](https://arxiv.org/abs/2103.06695) which is trained in a self-supervised manner to leverage audio augmentation methods. Generating robust speech representations invariant to minimal differences in audio.

BYOL-S model was pretrained on a subset of [AudioSet](https://research.google.com/audioset/) that is related to speech only. We further modified the BYOL-S network to learn from handcrafted (openSMILE) features and data-driven features engendering [Hybrid BYOL-S](https://arxiv.org/abs/2203.16637). Hybrid BYOL-S outperformed its predecessor BYOL-S in most tasks encountered in the [HEAR Competition 2021](https://neuralaudio.ai/).

In this repo, we provide the weights for [BYOL-S](https://arxiv.org/abs/2110.03414) and [Hybrid BYOL-S](https://arxiv.org/abs/2203.16637). In addition to a package called *serab-byols* to facilitate generating these speech representations.


### Demo
* A quick demo demonstrating the extraction of Hybrid BYOL-S embeddings on a Colab notebook is available [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tvL-_rAY6uYPGLrcdSFJoaFG1mkOZkud#scrollTo=_MMd-YD6wke6)


### Installation

Tested with Python 3.7 and 3.8.

**Method: pip local source tree**

```python
git clone https://github.com/GasserElbanna/serab-byols.git
python3 -m pip install -e ./serab-byols
```

### (Hybrid) BYOL-S Model

The BYOL-S model inputs log-scaled Mel-frequency spectrograms using a
64-band Mel filter. Each frame of the spectrogram is then projected to 2048
dimensions using pretrained encoder. Weights for the projection matrix were
generated by training the BYOL-S network and are stored in this repository in the
directory `checkpoints`.

### Encoders:

The (Hybrid) BYOL-S model has been trained with different encoder architectures:
* AudioNTT: Original encoder used in [BYOL-A](https://arxiv.org/abs/2103.06695)
* Resnetish34: Adapted from this [repo](https://github.com/daisukelab/sound-clf-pytorch/blob/master/src/models.py)
* CLSTM: Inspired from this [paper](https://www.degruyter.com/document/doi/10.1515/jisys-2018-0372/html?lang=de#j_jisys-2018-0372_ref_030)
* CvT: Adapted from this [repo](https://github.com/lucidrains/vit-pytorch#cvt)

### Weights in the repo

* BYOL-S/AudioNTT: `checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth`
* BYOL-S/Resnetish34: `checkpoints/resnetish34_BYOLAs64x96-2105271915-e100-bs256-lr0003-rs42.pth`
* Hybrid BYOL-S/CvT (Best Model): `checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth`


### Usage

Audio embeddings can be computed using one of two methods: 1)
`get_scene_embeddings`, or 2) `get_timestamp_embeddings`.

`get_scene_embeddings` accepts a batch of audio clips (list of torch tensors) and generates a single embedding
for each audio clip. This can be computed as shown below:

```python
import torch
import serab_byols

model_name = 'cvt'
checkpoint_path = "serab-byols/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth"
# Load model with weights - located in the root directory of this repo
model = serab_byols.load_model(checkpoint_path, model_name)

# Create a batch of 2 white noise clips that are 2-seconds long as a dummy example
# and compute scene embeddings for each clip
audio = torch.rand((2, model.sample_rate * 2))
embeddings = serab_byols.get_scene_embeddings(audio, model)
```

The `get_timestamp_embeddings` method works exactly the same but returns an array
of embeddings from audio segment computed every 50ms (could be changed) over the duration of the input audio. An array
of timestamps corresponding to each embedding is also returned.

```python
import torch
import serab_byols

model_name = 'default'
checkpoint_path = 'serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth'
# Load model with weights - located in the root directory of this repo
model = serab_byols.load_model(checkpoint_path, model_name)

# Create a batch of 2 white noise clips that are 2-seconds long as a dummy example
# and compute scene embeddings for each clip
frame_duration = 1000 #ms
hop_size = 50 #ms
audio = torch.rand((2, model.sample_rate * 2))
embeddings, timestamps = serab_byols.get_timestamp_embeddings(audio, model, frame_duration, hop_size)
```

NOTE: All BYOL-S variants were pretrained on audios with sampling rate 16kHz. Make sure to resample your dataset to 16kHz to be compatible with the model's requirements.

### Citations

If you are using this package please cite the [paper](https://arxiv.org/abs/2203.16637):

```python
@article{elbanna2022hybrid,
  title={Hybrid Handcrafted and Learnable Audio Representation for Analysis of Speech Under Cognitive and Physical Load},
  author={Elbanna, Gasser and Biryukov, Alice and Scheidwasser-Clow, Neil and Orlandic, Lara and Mainar, Pablo and Kegler, Mikolaj and Beckmann, Pierre and Cernak, Milos},
  journal={arXiv preprint arXiv:2203.16637},
  year={2022}
}
```
