**[This README adapted to my modification of the WaveGlow implementation]**

![WaveGlow](waveglow_logo.png "WaveGLow")

## WaveGlow: a Flow-based Generative Network for Speech Synthesis

### Ryan Prenger, Rafael Valle, and Bryan Catanzaro

```
In our recent [paper], we propose WaveGlow: a flow-based network capable of
generating high quality speech from mel-spectrograms. WaveGlow combines insights
from [Glow] and [WaveNet] in order to provide fast, efficient and high-quality
audio synthesis, without the need for auto-regression. WaveGlow is implemented
using only a single network, trained using only a single cost function:
maximizing the likelihood of the training data, which makes the training
procedure simple and stable.

Our [PyTorch] implementation produces audio samples at a rate of 1200
kHz on an NVIDIA V100 GPU. Mean Opinion Scores show that it delivers audio
quality as good as the best publicly available WaveNet implementation.

Visit our [website] for audio samples.
```

## 1. Setup

1. Clone our repo and initialize submodule

   ```command
   git clone https://github.com/sungjae-cho/waveglow.git
   cd waveglow
   git submodule init
   git submodule update
   ```

2. Install requirements `pip3 install -r requirements.txt`

3. Install [Apex]


## 2. Generate audio with our pre-existing model

### 2.1. From the command line interface
1. Download the [published model]. This is saved as  `pretrained/waveglow_256channels_universal_v5.pt`
2. Download [mel-spectrograms]. These are saved in `mel_spectrograms`
3. Generate audio `python3 inference.py -f <(ls mel_spectrograms/*.pt) -w pretrained/waveglow_256channels_universal_v5.pt -o mel_spectrograms. --is_fp16 -s 0.6`. Then, the command outputs are below.
```
mel_spectrograms/LJ001-0015.wav_synthesis.wav
mel_spectrograms/LJ001-0051.wav_synthesis.wav
mel_spectrograms/LJ001-0063.wav_synthesis.wav
mel_spectrograms/LJ001-0072.wav_synthesis.wav
mel_spectrograms/LJ001-0079.wav_synthesis.wav
mel_spectrograms/LJ001-0094.wav_synthesis.wav
mel_spectrograms/LJ001-0096.wav_synthesis.wav
mel_spectrograms/LJ001-0102.wav_synthesis.wav
mel_spectrograms/LJ001-0153.wav_synthesis.wav
mel_spectrograms/LJ001-0173.wav_synthesis.wav
```

N.b. use `convert_model.py` to convert your older models to the current model
with fused residual and skip connections.

### 2.2. From the python script

```python
import torch
from scipy.io.wavfile import write
from denoiser import Denoiser

sampling_rate = 22050
audio_path = 'audio.wav'
waveglow_path = new_args.init_waveglow
waveglow = torch.load(waveglow_path)['model']
for k, m in waveglow.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda().eval()
wg_denoiser = Denoiser(waveglow).cuda()
wg_denoiser_strength = 0.1 # Removes model bias. Start with 0.1 and adjust
wg_sigma = 0.85

audio = wg_denoiser(waveglow.infer(mel_outputs_postnet, sigma=wg_sigma), wg_denoiser_strength)

audio = audio.squeeze().cpu().numpy()
maxv = 2 ** (16 - 1)
audio /= max(abs(audio.max()), abs(audio.min()))
audio = (audio * maxv * 0.95).astype(np.int16)

write(audio_path, sampling_rate, audio)
```

## 3. Train your own model

### 3.1. Train your own model

1. Save a list of the file names for training into `train_files.txt` and for testing into `test_files.txt`.

3. Train your WaveGlow networks

   ```command
   python train.py -c config.json --prj_name prj_name --run_name run_name --visible_gpus 1
   ```

   For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.

   ```command
   python distributed.py -c config.json --prj_name prj_name --run_name run_name --visible_gpus 1,2,3
   ```

   For mixed precision training set `"fp16_run": true` on `config.json`.

### 3.2. Test your own model

1. Make test set mel-spectrograms

   `python mel2samp.py -f test_files.txt -o . -c config.json`

2. Do inference with your network at iteration 10000 of the  `checkpoints/prj_name/run_name/waveglow_10000` checkpoint.

   ```command
   ls *.pt > mel_files.txt
   python3 inference.py -f mel_files.txt -w checkpoints/prj_name/run_name/waveglow_10000 -o . --is_fp16 -s 0.6
   ```

[//]: # (TODO)
[//]: # (PROVIDE INSTRUCTIONS FOR DOWNLOADING LJS)
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[paper]: https://arxiv.org/abs/1811.00002
[WaveNet implementation]: https://github.com/r9y9/wavenet_vocoder
[Glow]: https://blog.openai.com/glow/
[WaveNet]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[PyTorch]: http://pytorch.org
[published model]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[mel-spectrograms]: https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
