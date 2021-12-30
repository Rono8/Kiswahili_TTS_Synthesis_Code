# Kiswahili_TTS_Synthesis_Code
The synthesis code for the Kiswahili TTS is
```
import os
import sys
! git clone https://github.com/Rono8/kiswahili_tts
os.chdir("./kiswahili_tts")
!pip install .
!pip install git+https://github.com/repodiac/german_transliterate.git #egg=german_transliterate
!pip uninstall tensorflow -y
!pip install tensorflow==2.3
import itertools
import logging
import os
import random
import argparse
import logging

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow as tf
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

import IPython.display as ipd
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)
# Download pretrained  model zip file from drive
print("Downloading zip model...")
!gdown --id {"1j7IHz_2m60vVhxgp9VK3EXd2JlKz2xk3"} -O model-150000.h5
#!gdown --id {"1-1cpmelBQG2VaptQk8bFYmrUyhMYyivL"} -O tacotron2.v1.yaml

# Download pretrained Vocoder model
print("Downloading MelGAN model...")
!gdown --id {"1A3zJwzlXEpu_jHeatlMdyPGjn1V7-9iG"} -O melgan-1M6.h5
!gdown --id {"1Ys-twSd3m2uqhJOEiobNox6RNQf4txZs"} -O melgan_config.yml

# Load Vocoder
melgan_config = AutoConfig.from_pretrained('/content/kiswahili_tts/examples/melgan/conf/melgan.v1.yaml')
melgan = TFAutoModel.from_pretrained(
    config=melgan_config,
   pretrained_path="melgan-1M6.h5",
    name="melgan"
)

# Load Model
tacotron2_config = AutoConfig.from_pretrained('/content/kiswahili_tts/examples/tacotron2/conf/tacotron2.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    
    pretrained_path="/content/kiswahili_tts/model-150000.h5",
    name="tacotron2"
)

#processor = AutoProcessor.from_pretrained(pretrained_path="/content/drive/MyDrive/ljspeech_mapper.json")
processor = AutoProcessor.from_pretrained(pretrained_path="/content/kiswahili_tts/tensorflow_tts/ljspeech_mapper.json")
def do_synthesis(input_text, text2mel_model, vocoder_model):
    input_ids = processor.text_to_sequence(input_text)

    # text2mel part
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )

    # vocoder part
    audio = vocoder_model(mel_outputs)[0, :, 0]

    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()

def visualize_attention(alignment_history):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()

def visualize_mel_spectrogram(mels):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()
break
text = input('Input the Kiswahili text:')
print(text)
mels, alignment_history, audios = do_synthesis(text, tacotron2, melgan)
visualize_attention(alignment_history[0])
visualize_mel_spectrogram(mels[0])
ipd.Audio(audios, rate=22050)

```
