# GenAI Text Generation Example

This example demonstrates basic text generation using ONNX Runtime GenAI with the Phi-3-mini model.

## Prerequisites

1. **ONNX Runtime GenAI Library**

   Download from [ONNX Runtime GenAI Releases](https://github.com/microsoft/onnxruntime-genai/releases):
   - macOS: `libonnxruntime-genai.dylib`
   - Linux: `libonnxruntime-genai.so`
   - Windows: `onnxruntime-genai.dll`

2. **Phi-3-mini Model**

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

## Environment Variables

- `ONNXRUNTIME_GENAI_LIB_PATH`: Path to ONNX Runtime GenAI shared library (required)

## Expected Output

```
Loading model: ./models/phi3-mini/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
Prompt tokens: 7

--- Generation Output ---
What is the capital of France? The capital of France is Paris.
--- End of Generation ---
```

## Troubleshooting

1. **Library not found**: Ensure `ONNXRUNTIME_GENAI_LIB_PATH` is set correctly and the library is accessible.

2. **Model loading fails**: Verify the model directory contains `genai_config.json` and the ONNX model files.

3. **Slow generation**: The INT4 CPU model runs at ~20 tokens/sec. Consider using GPU acceleration for faster inference.

4. **macOS ARM64 crash on library load**: The pre-built macOS ARM64 binaries have a known issue where they crash during static initialization with `std::runtime_error` in `Ort::InitApi()`. This is a problem with the released binaries, not with the Go bindings.

   **Workarounds:**
   - Build ONNX Runtime GenAI from source for macOS ARM64
   - Use Linux x64 environment for testing
   - Check [GitHub Issues](https://github.com/microsoft/onnxruntime-genai/issues) for updates

   **Debug output:**
   ```
   libc++abi: terminating due to uncaught exception of type std::runtime_error
   SIGABRT: abort
   ```

   This crash occurs in:
   ```
   libonnxruntime-genai.dylib`Ort::InitApi()
   ```
