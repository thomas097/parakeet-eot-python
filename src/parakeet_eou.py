import math
import numpy as np
from scipy.fft import fft
from collections import deque
from numpy.typing import NDArray
from .model_eou import EOUModel, EncoderCache
from .tokenizer import ParakeetTokenizer

from .utils import Timer

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 5.9604645e-8
FMAX = 8000.0

class ParakeetEOUModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        vocab_size = tokenizer.get_vocab_size()
        self.blank_id = vocab_size - 1 if vocab_size - 1 >= 1000 else 1026
        self.eou_id = tokenizer.token_to_id("<EOU>")

        self.encoder_cache = EncoderCache()
        self.state_h = np.zeros((1, 1, 640), dtype=np.float32)
        self.state_c = np.zeros((1, 1, 640), dtype=np.float32)
        self.last_token = np.full((1, 1), self.blank_id, dtype=np.int32)
        self.last_non_blank_token = None

        self.mel_basis = self.create_mel_filterbank()
        self.window = self.create_window()
        self.buffer_size_samples = SAMPLE_RATE * 4  # 4-second buffer
        self.audio_buffer = deque(maxlen=self.buffer_size_samples)

    @classmethod
    def from_pretrained(cls, path: str) -> 'ParakeetEOUModel':
        tokenizer = ParakeetTokenizer.from_pretrained(path)
        model = EOUModel.from_pretrained(path)
        return cls(model=model, tokenizer=tokenizer)

    def transcribe(self, chunk: NDArray) -> str:
        self.audio_buffer.extend(chunk.flatten())

        MIN_BUFFER_SAMPLES = SAMPLE_RATE  # 1 second
        if len(self.audio_buffer) < MIN_BUFFER_SAMPLES:
            return ""

        audio_data = np.array(self.audio_buffer, dtype=np.float32)
        full_features = self.extract_mel_features(audio_data)
        total_frames = full_features.shape[2]

        PRE_ENCODE_CACHE = 9
        FRAMES_PER_CHUNK = 16
        SLICE_LEN = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK
        start_frame = max(0, total_frames - SLICE_LEN)

        features = full_features[:, :, start_frame:]
        time_steps = features.shape[2]

        with Timer("model.run_encoder"):
            encoder_out, self.encoder_cache = self.model.run_encoder(features, time_steps, self.encoder_cache)

        total_frames = encoder_out.shape[2]
        if total_frames == 0:
            return ""

        text_output = ""
        for t in range(total_frames):
            current_frame = encoder_out[:, :, t:t+1]

            syms_added = 0
            while syms_added < 5:
                with Timer("model.run_decoder"):
                    logits, new_h, new_c = self.model.run_decoder(
                        current_frame, self.last_token, self.state_h, self.state_c
                    )
                vocab = logits[0, 0, :]
                max_idx = int(np.argmax(np.where(np.isfinite(vocab), vocab, -np.inf)))

                if max_idx == self.eou_id:
                    if self.last_non_blank_token == self.eou_id:
                        break
                    self.last_non_blank_token = self.eou_id
                    return text_output + " [EOU]"

                if max_idx in (self.blank_id, 0):
                    break
                if max_idx >= self.tokenizer.get_vocab_size():
                    break

                self.state_h = new_h
                self.state_c = new_c
                self.last_token.fill(max_idx)
                self.last_non_blank_token = max_idx

                token = self.tokenizer.id_to_token(max_idx)
                if token:
                    text_output += token.replace('▁', ' ')
                syms_added += 1

        return text_output

    def reset_states(self):
        self.state_h.fill(0.0)
        self.state_c.fill(0.0)
        self.last_token.fill(self.blank_id)

    def extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        with Timer("extract_mel_features"):
            audio_pre = self.apply_preemphasis(audio)
            spec = self.stft(audio_pre)
            mel = self.mel_basis @ spec
            mel_log = np.log(np.maximum(mel, 0.0) + LOG_ZERO_GUARD)
            return mel_log[np.newaxis, :, :]  # add channel axis

    @staticmethod
    def apply_preemphasis(audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio
        result = np.empty_like(audio)
        result[0] = audio[0]
        result[1:] = audio[1:] - PREEMPH * audio[:-1]
        result[~np.isfinite(result)] = 0.0
        return result

    def stft(self, audio: np.ndarray) -> np.ndarray:
        with Timer("extract_mel_features"):
            pad_amount = N_FFT // 2
            padded_audio = np.pad(audio, (pad_amount, pad_amount), mode='constant')
            num_frames = 1 + (len(padded_audio) - WIN_LENGTH) // HOP_LENGTH
            freq_bins = N_FFT // 2 + 1
            spec = np.zeros((freq_bins, num_frames), dtype=np.float32)

            for frame_idx in range(num_frames):
                start = frame_idx * HOP_LENGTH
                if start + WIN_LENGTH > len(padded_audio):
                    break
                windowed = padded_audio[start:start+WIN_LENGTH] * np.array(self.window)
                fft_frame = fft(np.pad(windowed, (0, N_FFT - WIN_LENGTH)))
                mag_sq = np.absolute(fft_frame[:freq_bins]) ** 2 #type:ignore
                spec[:, frame_idx] = np.where(np.isfinite(mag_sq), mag_sq, 0.0)
            return spec

    @staticmethod
    def create_window() -> np.ndarray:
        with Timer("create_window"):
            return np.array([
                0.5 - 0.5 * math.cos(2.0 * math.pi * i / (WIN_LENGTH - 1))
                for i in range(WIN_LENGTH)
            ], dtype=np.float32)

    @staticmethod
    def create_mel_filterbank() -> np.ndarray:
        with Timer("create_mel_filterbank"):
            num_freqs = N_FFT // 2 + 1
            hz_to_mel = lambda hz: 2595.0 * math.log10(1 + hz / 700.0)
            mel_to_hz = lambda mel: 700.0 * (10**(mel / 2595.0) - 1.0)
            mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(FMAX)
            mel_points = [mel_to_hz(mel_min + (mel_max - mel_min) * i / (N_MELS + 1))
                        for i in range(N_MELS + 2)]
            fft_freqs = [(SAMPLE_RATE / N_FFT) * i for i in range(num_freqs)]
            weights = np.zeros((N_MELS, num_freqs), dtype=np.float32)

            for i in range(N_MELS):
                left, center, right = mel_points[i], mel_points[i+1], mel_points[i+2]
                for j, freq in enumerate(fft_freqs):
                    if left <= freq <= center:
                        weights[i, j] = (freq - left) / (center - left)
                    elif center < freq <= right:
                        weights[i, j] = (right - freq) / (right - center)
                weights[i, :] *= 2.0 / (right - left)  # normalize
            return weights
