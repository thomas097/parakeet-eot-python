# 🐦 Pyrakeet

An optimized Python implementation of the **Parakeet Realtime EOU-120M** streaming ASR model by NVIDIA.

Pyrakeet converts the original model to **ONNX** to enable cross-platform deployment and supports **UInt8 quantization** for ultra-low-latency, real-time speech recognition — facilitating efficient inference, even on low-resource, CPU-only devices.


## 📌 Overview

This repository provides a streamlined Python implementation for running **NVIDIA Parakeet Realtime EOU-120M v1** using **ONNX Runtime**.

The main goals of this project are:

- 🔄 ONNX conversion of NVIDIA’s _Parakeet Realtime EOU 120M v1_ model.  
- ⚡ UInt8 quantization to reduce latency, improve throughput, and minimize memory usage  
- 🧩 A fully reimplemented preprocessing and inference pipeline with **no PyTorch or NVIDIA NeMo runtime dependencies**
- 🎮 CUDA support (for non-quantized model only)  
- ✋ Improve the model's built-in end-of-utterance (EOU) detection by modifying the decoding strategy  

This makes Pyrakeet ideal for local voice assistants and interactive applications. 


## 🚀 Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/thomas097/parakeet-eot-python.git
cd parakeet-eot-python
````

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download non-quantized models:

```bash
cd checkpoints/parakeet-eou
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/decoder_joint.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/encoder.onnx
wget https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/tokenizer.json
```

5. UInt8 quantization (optional, but recommended for CPU deployment)
```
python scripts/quantize_onnx_partial_uint8.py
```

When prompted to provide a model path, specify the relative path from the root of the project to the model file. For example, `checkpoints/parakeet-eou/encoder.onnx`.



### 🧪 Dependencies

The project has been tested with the following dependencies:

```bash
tokenizers==0.19.1
sounddevice==0.5.1
numpy==1.25.2
scipy==1.10.1
onnxruntime==1.19.2

# Only required for uint8 quantization
onnx==1.20.0
onnxruntime-tools==1.7.0
```

Using these versions is recommended for reproducible results.



## ▶️ Usage

### Basic Example

```bash
python example.py
```

This will:

* 🎙️ Capture audio from the default microphone
* 🔊 Perform streaming ASR
* 📝 Emit transcriptions when an end-of-utterance is detected



### Advanced Usage

```python
from src import ParakeetEOUModel, AudioBuffer, AudioRecorder

# Load quantized model and tokenizer
parakeet = ParakeetEOUModel.from_pretrained(
    path="checkpoints/parakeet-eou",
    device="cpu",
    quant="uint8"
)

# Load audio sampled at 16kHz as chunks of 160ms (2560 samples per chunk)
audio = ...

for chunk in audio:
    new_tokens = parakeet.transcribe(chunk)
    print(new_tokens)
```

The model maintains its internal state automatically — no need to manage it explicitly when calling `.transcribe()`.



## 🙏 Attribution

* **NVIDIA** — Parakeet Realtime EOU-120M model and research
* **ONNX Runtime** — High-performance inference engine

All rights to the original, full-precision *Parakeet EOU 120M v1* model belong to NVIDIA.


## 📄 License

The source code is distributed under the **Apache 2.0 License**.

The Parakeet EOU 120M model itself is governed by NVIDIA’s Open Model licensing terms.
For details, see the LICENSE file or visit the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).