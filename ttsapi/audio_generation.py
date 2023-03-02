import base64
import io
import os
from typing import Tuple

import requests
import torch
import torchaudio
import yaml


class SpeechGenerator:
    def __init__(self,
                 tts_url: str = "https://65n1fnjut7.execute-api.eu-west-1.amazonaws.com/dev/v1/tts-api-manager",
                 token_header_key: str = "x-api-key",
                 token_variable_name: str = "TTS_TOKEN",
                 sample_rate: int = 16000,
                 content_type: str = "application/json"):
        self.tts_url = tts_url

        access_token = os.environ.get(token_variable_name)

        self.request_headers = {token_header_key: access_token,
                                "Content-Type": content_type,
                                "sample-rate": str(sample_rate)}

    def send_request(self, text: str, language_symbol: str) -> requests.Response:
        contents = {"text": text,
                    "as_url": False,
                    "lang": language_symbol}
        return requests.post(url=self.tts_url,
                             json=contents,
                             headers=self.request_headers)

    def _decode_response(self, response: requests.Response) -> bytes:
        response_dict = response.json()
        audio = response_dict["audio"]['audio_data']
        return base64.b64decode(audio)

    def _read_audio_bytes(self, audio_bytes: bytes) -> Tuple[torch.Tensor, int]:
        buffered_audio = io.BytesIO(audio_bytes)
        y, sr = torchaudio.load(buffered_audio)
        y = y.detach().cpu()
        return y, sr

    def generate_audio(self, text: str, language_symbol: str) -> Tuple[torch.Tensor, int]:
        response = self.send_request(text, language_symbol)
        audio_bytes = self._decode_response(response)
        audio_tensor, sample_rate = self._read_audio_bytes(audio_bytes)
        return audio_tensor, sample_rate

    def __call__(self, text: str, language_symbol: str) -> Tuple[torch.Tensor, int]:
        return self.generate_audio(text, language_symbol)


def load_speech_generator(configs_file_path: str = "configs.yaml"):
    with open(configs_file_path, "r") as f:
        configs_dict = yaml.safe_load(f)
    return SpeechGenerator(**configs_dict['speech_generation'])
