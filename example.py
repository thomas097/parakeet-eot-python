from src import ParakeetEOUModel, AudioBuffer, AudioRecorder

# Load model and tokenizer
parakeet = ParakeetEOUModel.from_pretrained(
    path="checkpoints/parakeet-eou",
    device="cpu",
    quant="uint8"
    )

# Prepare recording device
buffer = AudioBuffer()
recorder = AudioRecorder(
    buffer=buffer,
    samplerate=16_000,
    channels=1,
    dtype="float32",
    chunk_size=2560 # 160ms
    )
recorder.start()

# Process in 160ms chunks for streaming
text_output = ""
while recorder.is_recording():
    chunks = buffer.get_contents(clear=True)
    for chunk in chunks:
        text = parakeet.transcribe(chunk)

        print(text, end="", flush=True)
        text_output += text

        if "[EOU]" in text:
            print()

        if "stop" in text:
            break

recorder.stop()
