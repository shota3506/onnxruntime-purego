# GenAI Text Generation Example

This example demonstrates basic text generation using ONNX Runtime GenAI with the Phi-3-mini model.

See [examples/README.md](../README.md) for common GenAI prerequisites and troubleshooting.

## Prerequisites

1. **Phi-3-mini Model**

   Download using Hugging Face CLI:
   ```bash
   hf download microsoft/Phi-3-mini-4k-instruct-onnx \
     --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" \
     --local-dir models/phi3-mini
   ```

   This downloads the INT4 quantized CPU version (~2GB).

## Usage

```bash
# Build the example
go build -o genai-example ./examples/genai/

# Set environment variable
export ONNXRUNTIME_GENAI_LIB_PATH=/path/to/libonnxruntime-genai.dylib

# Run with a prompt
./genai-example \
  -model ./models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 \
  "What is the capital of France?"

# Custom max length
./genai-example \
  -model ./models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 \
  -max-length 300 \
  "Explain quantum computing in simple terms:"
```

## Options

- `-model`: Path to model directory containing `genai_config.json` (required)
- `-max-length`: Maximum tokens to generate (default: 200)

## Expected Output

```
Loading model: ./models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
Prompt tokens: 7

--- Generation Output ---
What is the capital of France? The capital of France is Paris.
--- End of Generation ---
```

## Troubleshooting

1. **Model loading fails**: Verify the model directory contains `genai_config.json` and the ONNX model files.

2. **Slow generation**: The INT4 CPU model runs at ~20 tokens/sec. Consider using GPU acceleration for faster inference.
