from typing import Optional, Tuple

import torch

from .audio_generation import SpeechGenerator, load_speech_generator
from .audutils import SpeechEnhancer, load_speech_enhancer


class TextToSpeech:
    def __init__(self, configs_file_path: str,
                 speech_generator: Optional[SpeechGenerator] = None,
                 speech_enhancer: Optional[SpeechEnhancer] = None,
                 ):
        self.speech_enhancer = speech_enhancer or load_speech_enhancer(configs_file_path)
        self.speech_generator = speech_generator or load_speech_generator(configs_file_path)

    def _generate_speech(self, text: str, language_symbol: str) -> Tuple[torch.Tensor, int]:
        return self.speech_generator(text, language_symbol)

    def _enhance_speech(self, speech_tensor: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        return self.speech_enhancer(speech_tensor, sample_rate)

    def generate_enhanced_speech(self, text: str,
                                 language_symbol: str,
                                 apply_enhancement: bool) -> Tuple[torch.Tensor, int]:
        audio_tensor, sample_rate = self._generate_speech(text, language_symbol)

        if apply_enhancement:
            audio_tensor, sample_rate = self._enhance_speech(audio_tensor, sample_rate)

        return audio_tensor, sample_rate

    def __call__(self, text: str, language_symbol: str,
                 apply_enhancement: bool) -> Tuple[torch.Tensor, int]:
        return self.generate_enhanced_speech(text, language_symbol, apply_enhancement)
