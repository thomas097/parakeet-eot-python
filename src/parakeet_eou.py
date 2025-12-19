import math
import numpy as np
from scipy.fft import fft, rfft
from numpy.lib.stride_tricks import as_strided
from collections import deque
from numpy.typing import NDArray
from .model_eou import EOUModel, EncoderCache
from .tokenizer import ParakeetTokenizer

from .utils import Timer

SAMPLE_RATE = 16000
MIN_BUFFER_SIZE = 6
PRE_ENCODE_CACHE = 9
FRAMES_PER_CHUNK = 16
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 128
PREEMPH = 0.97
LOG_ZERO_GUARD = 5.9604645e-8
FMAX = 8000.0

class ParakeetEOUModel:
    def __init__(self, model: EOUModel, tokenizer: ParakeetTokenizer):
        self._model = model
        self._tokenizer = tokenizer

        self._blank_id = tokenizer.token_to_id("<EOB>")
        self._eou_id = tokenizer.token_to_id("<EOU>")

        self._encoder_cache = EncoderCache()
        self._state_h = np.zeros((1, 1, 640), dtype=np.float32)
        self._state_c = np.zeros((1, 1, 640), dtype=np.float32)
        self._last_token = np.full((1, 1), self._blank_id, dtype=np.int32)
        self._last_non_blank_token = None

        self._mel_basis = self._create_mel_filterbank()
        self._window = np.hanning(WIN_LENGTH).astype(np.float32)
        self._audio_buffer = deque(maxlen=SAMPLE_RATE * MIN_BUFFER_SIZE)

    @classmethod
    def from_pretrained(cls, path: str) -> 'ParakeetEOUModel':
        tokenizer = ParakeetTokenizer.from_pretrained(path)
        model = EOUModel.from_pretrained(path)
        return cls(model=model, tokenizer=tokenizer)

    def transcribe(self, chunk: NDArray) -> str:
        self._audio_buffer.extend(chunk.flatten())

        if self._audio_buffer.maxlen is not None and len(self._audio_buffer) < self._audio_buffer.maxlen:
            return ""

        audio_data = np.array(self._audio_buffer, dtype=np.float32)
        full_features = self.extract_mel_features(audio_data)
        total_frames = full_features.shape[2]
        start_frame = max(0, total_frames - PRE_ENCODE_CACHE - FRAMES_PER_CHUNK)

        features = full_features[:, :, start_frame:]
        time_steps = features.shape[2]

        with Timer("model.run_encoder"):
            encoder_out, self._encoder_cache = self._model.run_encoder(
                features=features, 
                length=time_steps, 
                cache=self._encoder_cache
                )

        total_frames = encoder_out.shape[2]
        if total_frames == 0:
            return ""

        text_output = ""
        for t in range(total_frames):
            current_frame = encoder_out[:, :, t:t+1]

            syms_added = 0
            while syms_added < 5:
                with Timer("model.run_decoder"):
                    logits, new_h, new_c = self._model.run_decoder(
                        encoder_frame=current_frame, 
                        last_token=self._last_token, 
                        state_h=self._state_h, 
                        state_c=self._state_c
                    )

                vocab = logits[0, 0, :]
                max_idx = int(np.argmax(np.where(np.isfinite(vocab), vocab, -np.inf)))

                if max_idx == self._eou_id:
                    if self._last_non_blank_token == self._eou_id:
                        break
                    self._last_non_blank_token = self._eou_id
                    return text_output + " [EOU]"

                if max_idx in (self._blank_id, 0):
                    break
                if max_idx >= self._tokenizer.get_vocab_size():
                    break

                self._state_h = new_h
                self._state_c = new_c
                self._last_token.fill(max_idx)
                self._last_non_blank_token = max_idx

                token = self._tokenizer.id_to_token(max_idx)
                if token:
                    text_output += token.replace('▁', ' ')
                syms_added += 1

        return text_output

    def reset_states(self):
        self._state_h.fill(0.0)
        self._state_c.fill(0.0)
        self._last_token.fill(self._blank_id)

    def extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        with Timer("extract_mel_features"):
            audio_pre = self.apply_preemphasis(audio)
            mel = self._mel_basis @ self.stft(audio_pre)
            mel_log = np.log(np.maximum(mel, 0.0) + LOG_ZERO_GUARD)
            return mel_log[np.newaxis, :, :]

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
        with Timer("stft"):
            pad_amount = N_FFT // 2
            padded_audio = np.pad(audio, (pad_amount, pad_amount))

            num_frames = 1 + (len(padded_audio) - WIN_LENGTH) // HOP_LENGTH

            # Create strided frame view: shape (num_frames, WIN_LENGTH)
            frames = as_strided(
                padded_audio,
                shape=(num_frames, WIN_LENGTH),
                strides=(padded_audio.strides[0] * HOP_LENGTH,
                        padded_audio.strides[0]),
                writeable=False
            )

            # Windowing (window should be precomputed once as np.ndarray)
            windowed = frames * self._window

            # Zero-pad to N_FFT
            if WIN_LENGTH < N_FFT:
                pad_width = ((0, 0), (0, N_FFT - WIN_LENGTH))
                windowed = np.pad(windowed, pad_width)

            # Batched real FFT
            fft_frames = rfft(windowed, axis=1)

            # Power spectrum
            spec = np.abs(fft_frames) ** 2 #type:ignore

            # Transpose to match your original output shape
            return spec.T.astype(np.float32)

    @staticmethod
    def _create_mel_filterbank() -> np.ndarray:
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
