import os
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from numpy.typing import NDArray

class ModelError(Exception):
    pass

@dataclass
class EncoderCache:
    """
    Encoder state cache to maintain temporal context 
    across chunks during streaming inference.
    """
    # channel cache: [17, 1, 70, 512] - 17 layers, batch=1, 70 frame lookback
    cache_last_channel: NDArray = field(
        default_factory=lambda: np.zeros((17, 1, 70, 512), dtype=np.float32)
    )
    # time cache: [17, 1, 512, 8] - 17 layers, batch=1, fixed 8 time steps
    cache_last_time: NDArray = field(
        default_factory=lambda: np.zeros((17, 1, 512, 8), dtype=np.float32)
    )
    # cache length: [1]
    cache_last_channel_len: NDArray = field(
        default_factory=lambda: np.array([0], dtype=np.int64)
    )


class EOUModel:
    def __init__(
            self, 
            encoder_session: ort.InferenceSession, 
            decoder_session: ort.InferenceSession
            ):
        self.encoder = encoder_session
        self.decoder_joint = decoder_session

    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = 'cpu') -> 'EOUModel':
        """
        Convenience method to load an EOU model consisting of an encoder 
        and a joint decoder from an ONNX model directory.

        Args:
            model_dir (str): Path to the model files.
            device (str): Device to use for inference, e.g. 'cpu' (default) or 'cuda'.

        Raises:
            ModelError: When any of the required model files cannot be found.

        Returns:
            EOUModel: Instance of an EOU model.
        """
        encoder_path = os.path.join(model_dir, "encoder_int8.onnx")
        decoder_path = os.path.join(model_dir, "decoder_joint.onnx")

        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            raise ModelError(
                f"Missing ONNX files in {model_dir}. Expected encoder.onnx and decoder_joint.onnx"
            )
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        if device.lower().startswith('cuda'):
            providers.insert(0, "CUDAExecutionProvider")

        encoder_session = ort.InferenceSession(
            path_or_bytes=encoder_path, 
            providers=providers,
            sess_options=sess_options
            )

        decoder_session = ort.InferenceSession(
            path_or_bytes=decoder_path, 
            providers=providers,
            sess_options=sess_options
            )

        return cls(encoder_session, decoder_session)

    def run_encoder(
        self,
        features: NDArray,
        length: int,
        cache: EncoderCache
        ) -> tuple[NDArray, EncoderCache]:
        """Runs the stateful encoder with cache.

        Args:
            features (NDArray): Input audio features with shape [1, 128, T].
            length (int): Length of the input sequence.
            cache (EncoderCache): Encoder cache from the previous step.

        Returns:
            Tuple[NDArray, EncoderCache]:
                - Encoded features with shape [1, 512, T].
                - Updated encoder cache.
        """

        outputs = self.encoder.run(
            output_names=None,
            input_feed={
                "audio_signal": features.astype(np.float32, copy=False),
                "length": np.asarray([length], dtype=np.int64),
                "cache_last_channel": cache.cache_last_channel,
                "cache_last_time": cache.cache_last_time,
                "cache_last_channel_len": cache.cache_last_channel_len.astype(np.int64, copy=False),
            },
        )
        encoder_out, _, last_channel, last_time, last_channel_len = outputs

        cache = EncoderCache(
            cache_last_channel=last_channel,
            cache_last_time=last_time,
            cache_last_channel_len=last_channel_len,
        )

        return encoder_out, cache

    def run_decoder(
        self,
        encoder_frame: NDArray,
        last_token: NDArray,
        state_h: NDArray,
        state_c: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Runs the stateful decoder for a single step.

        Args:
            encoder_frame (NDArray): Encoder output frame with shape [1, 512, 1].
            last_token (NDArray): Previous token ID with shape [1, 1].
            state_h (NDArray): Previous decoder hidden state with shape [1, 1, 640].
            state_c (NDArray): Previous decoder cell state with shape [1, 1, 640].

        Returns:
            Tuple[NDArray, NDArray, NDArray]:
                - Logits with shape [1, 1, vocab].
                - Updated hidden state with shape [1, 1, 640].
                - Updated cell state with shape [1, 1, 640].
        """
        outputs = self.decoder_joint.run(
            output_names=None,
            input_feed={
                "encoder_outputs": encoder_frame.astype(np.float32, copy=False),
                "targets": last_token.astype(np.int32, copy=False),
                "target_length": np.array([1], dtype=np.int32),
                "input_states_1": state_h.astype(np.float32, copy=False),
                "input_states_2": state_c.astype(np.float32, copy=False),
            },
        )
        logits, _, new_h, new_c = outputs

        return logits, new_h, new_c
