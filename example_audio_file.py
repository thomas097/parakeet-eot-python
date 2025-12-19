import numpy as np
from src import ParakeetEOUModel
from scipy.io import wavfile

# Load model and tokenizer
parakeet = ParakeetEOUModel.from_pretrained("checkpoints/parakeet-eou", device='cpu')

# Prepare recording device
_, audio = wavfile.read("audio_samples/frankly_my_dear_16kHz.wav")
audio = audio.flatten().astype(np.float32) / np.max(np.absolute(audio))

# Process in 160ms chunks for streaming
result = parakeet.transcribe_audio(audio)
print(result)
