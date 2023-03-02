from typing import Tuple

import torch
import torchaudio
import yaml
from speechbrain.pretrained import SpectralMaskEnhancement


class SpeechEnhancer:
    def __init__(self,
                 pretrained_model: str = "speechbrain/metricgan-plus-voicebank",
                 save_pretrained_dir: str = r"./speech-enhancement-model",
                 device: str = "cuda",
                 volume_gain: float = 1.1,
                 fade_in_ratio: float = 0.30,
                 fade_out_ratio: float = 0.5,
                 fade_shape: str = 'linear',
                 gain_type: str = '-n',
                 pitch: int = 40,
                 new_sr: int = 48000,
                 tempo: float = 1.15):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SpectralMaskEnhancement.from_hparams(
            source=pretrained_model,
            savedir=save_pretrained_dir,
            run_opts={"device": self.device})

        self.volume_changer = torchaudio.transforms.Vol(gain=volume_gain, gain_type="amplitude")

        fade_in_steps = int(new_sr * fade_in_ratio)
        fade_out_steps = int(new_sr * fade_out_ratio)
        self.fade_in_out_applier = torchaudio.transforms.Fade(fade_in_len=fade_in_steps,
                                                              fade_out_len=fade_out_steps,
                                                              fade_shape=fade_shape)

        self.gain_type = gain_type
        self.pitch_modifier = pitch
        self.new_sr = new_sr
        self.tempo = tempo

        self.sox_effects = [
            ['gain', self.gain_type],
            ['pitch', str(self.pitch_modifier)],
            ['rate', str(self.new_sr)],
            ['tempo', str(self.tempo)],
        ]

    def _enhance_speech(self, speech_tensor: torch.Tensor) -> torch.Tensor:
        speech_tensor = speech_tensor.squeeze().to(self.device)
        return self.model.enhance_batch(speech_tensor, lengths=torch.tensor([1.]))

    def _apply_sox_effects(self, speech_tensor: torch.Tensor, sample_rate: int) -> Tuple:
        return torchaudio.sox_effects.apply_effects_tensor(speech_tensor,
                                                           effects=self.sox_effects,
                                                           sample_rate=sample_rate)

    def _apply_transforms(self, speech_tensor: torch.Tensor) -> torch.Tensor:
        speech_tensor = speech_tensor.cpu()
        speech_tensor = self.fade_in_out_applier(speech_tensor)
        return self.volume_changer(speech_tensor)

    @torch.inference_mode()
    def __call__(self, speech_tensor: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        speech_tensor = self._enhance_speech(speech_tensor)
        speech_tensor, sample_rate = self._apply_sox_effects(speech_tensor, sample_rate)
        speech_tensor = self._apply_transforms(speech_tensor)
        return speech_tensor, sample_rate


def load_speech_enhancer(configs_file_path: str = "configs.yaml"):
    with open(configs_file_path, "r") as f:
        configs_dict = yaml.safe_load(f)
    return SpeechEnhancer(**configs_dict['speech_enhancement'])
