# GenAI Vision Example - Phi-3.5-Vision

This example demonstrates multimodal vision-language processing using ONNX Runtime GenAI with the Phi-3.5-Vision model. The model can analyze images and answer questions about their content.

See [examples/README.md](../README.md) for common GenAI prerequisites and troubleshooting.

## Prerequisites

1. **Phi-3.5-Vision Model** (~3.2GB):
   ```bash
   huggingface-cli download microsoft/Phi-3.5-vision-instruct-onnx \
     --include "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*" \
     --local-dir models/phi3.5-vision
   ```

## Usage

```bash
# Build the example
go build -o phi3.5-vision-example ./examples/genai/phi3.5-vision/

# Set library path
export ONNXRUNTIME_GENAI_LIB_PATH=/path/to/libonnxruntime-genai.dylib

# Run with image(s)
./phi3.5-vision-example \
  -model ./models/phi3.5-vision/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 \
  -images ./image.jpg \
  -max-length 2048 \
  "What do you see in this image?"

# Multiple images: use comma-separated paths
# -images "./image1.jpg,./image2.jpg"
```

**Options:**
- `-model`: Path to model directory (required)
- `-images`: Image file path(s), comma-separated for multiple (required)
- `-max-length`: Maximum tokens to generate (default: 2048)

## Troubleshooting

1. **Program hangs or crashes at "Processing images..."**:

   The default configuration causes OOM (out-of-memory) errors on CPU due to memory arena allocation.

   Edit the downloaded `genai_config.json` to disable CPU memory arena.
   Add `"enable_cpu_mem_arena": false` to `session_options` sections.

   Reference: [GitHub Issue #1146](https://github.com/microsoft/onnxruntime-genai/issues/1146)

2. **Model loading fails**: Verify the model directory contains all required files (genai_config.json, processor_config.json, and three ONNX models).

3. **Out of memory during generation**: Reduce `-max-length` or close other applications.
