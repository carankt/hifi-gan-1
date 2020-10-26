import torch
import numpy as np
from scipy.io.wavfile import write
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav

def main():
    mel_path = "/results/mels_and_audios/FastSpeech2LJ/FS2_tokenwise_v3_para"
    vocoder = torch.jit.load('/results/chkpts/LJ/Hifi-GAN/original/hifigan_pre_trained_v1.pt').cpu()
    vocoder.eval()
    mel = np.load(f"{mel_path}/mel_0.npy")
    mel = torch.from_numpy(mel)
    m = mel.unsqueeze(0)
    m = m.cpu()
    wav = vocoder(m)
    wav = wav.squeeze()
    wav = wav * MAX_WAV_VALUE
    wav = wav.detach().cpu().numpy().astype('int16')

    write(f"{mel_path}/mel_0_hifigan.wav", 22050, wav)
if __name__ == "__main__":
    main()
