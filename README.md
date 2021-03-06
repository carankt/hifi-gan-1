# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

### Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

In our [paper](https://arxiv.org/abs/2010.05646), 
we proposed HiFi-GAN: a GAN-based model capable of generating high fidelity speech efficiently.<br/>
We provide our implementation and pretrained models as open source in this repository.

**Abstract :**
Several recent work on speech synthesis have employed generative adversarial networks (GANs) to produce raw waveforms. 
Although such methods improve the sampling efficiency and memory usage, 
their sample quality has not yet reached that of autoregressive and flow-based generative models. 
In this work, we propose HiFi-GAN, which achieves both efficient and high-fidelity speech synthesis. 
As speech audio consists of sinusoidal signals with various periods, 
we demonstrate that modeling periodic patterns of an audio is crucial for enhancing sample quality. 
A subjective human evaluation (mean opinion score, MOS) of a single speaker dataset indicates that our proposed method 
demonstrates similarity to human quality while generating 22.05 kHz high-fidelity audio 167.9 times faster than 
real-time on a single V100 GPU. We further show the generality of HiFi-GAN to the mel-spectrogram inversion of unseen 
speakers and end-to-end speech synthesis. Finally, a small footprint version of HiFi-GAN generates samples 13.4 times 
faster than real-time on CPU with comparable quality to an autoregressive counterpart.

Visit our [demo website](https://jik876.github.io/hifi-gan-demo/) for audio samples.


## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config config_v1.json
```
To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.


## Pretrained Model
You can also use pretrained models we provide.<br/>
[Download pretrained models](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)<br/> 
Details of each folder are as in follows:

|Folder Name|Generator|Dataset|Fine-Tuned|
|------|---|---|---|
|LJ_V1|V1|LJSpeech|No|
|LJ_V2|V2|LJSpeech|No|
|LJ_V3|V3|LJSpeech|No|
|LJ_FT_T2_V1|V1|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V2|V2|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|LJ_FT_T2_V3|V3|LJSpeech|Yes ([Tacotron2](https://github.com/NVIDIA/tacotron2))|
|VCTK_V1|V1|VCTK|No|
|VCTK_V2|V2|VCTK|No|
|VCTK_V3|V3|VCTK|No|
|UNIVERSAL_V1|V1|Universal|No|


## Inference from wav file
1. Make `test_files` directory and copy wav files into the directory.
2. Run the following command.
```
python inference.py --checkpoint_file [generator checkpoint file path]
```
Generated wav files are saved in `generated_files` by default.<br>
You can change the path by adding `--output_dir` option.


## Inference for end-to-end speech synthesis
1. Make `test_mel_files` directory and copy generated mel-spectrogram files into the directory.<br>
You can generate mel-spectrograms using [Tacotron2](https://github.com/NVIDIA/tacotron2), 
[Glow-TTS](https://github.com/jaywalnut310/glow-tts) and so forth.
2. Run the following command.
```
python inference_e2e.py --checkpoint_file [generator checkpoint file path]
```
Generated wav files are saved in `generated_files_from_mel` by default.<br>
You can change the path by adding `--output_dir` option.


## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

# Additional feature 

Export the pre-trained model using torchscript
```
python export_torchscript.py -p /path/to/chkpt -c /path/to/config -i any_mel.npy --out /path/for/output_pt_file/ 
```
### For inference using torchscript model use the following code

```
import torch
import numpy as np
from scipy.io.wavfile import write
MAX_WAV_VALUE = 32768.0

vocoder = torch.jit.load('/path/to/pre-trained_torchscript.pt').cpu()  # load the jit file from path
vocoder.eval()
mel = np.load("/path/to/mel.npy")
mel = torch.from_numpy(mel)
m = mel.unsqueeze(0).cpu()
wav = vocoder(m)
wav = wav.squeeze()
wav = wav * MAX_WAV_VALUE
wav = wav.detach().cpu().numpy().astype('int16')

write(f"{mel_path}/mel_0_hifigan.wav", 22050, wav)
```

### Running the model for fine tuning on custom dataset
```sudo python3 train.py --fine_tuning True --config config_v1.json --input_wavs_dir /mnt/Karan/jared_final/wavs_trail_sil_trim/ --input_mels_dir /mnt/Karan/jared_final/preprocessed_trail_trim/gta --input_training_file /mnt/Karan/jared_final/filelists/raw_trail_trim_nopausil_eol/train.txt --input_validation_file /mnt/Karan/jared_final/filelists/raw_trail_trim_nopausil_eol/valid.txt```





