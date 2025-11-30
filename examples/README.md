# Examples

This directory contains example applications demonstrating the usage of onnxruntime-purego.

## ONNX Runtime Examples

- [**resnet**](./resnet/): Image classification using ResNet
- [**roberta-sentiment**](./roberta-sentiment/): Sentiment analysis using RoBERTa
- [**yolov10**](./yolov10/): Object detection using YOLOv10

## ONNX Runtime GenAI Examples

- [**genai/phi3**](./genai/phi3/): Text generation using Phi-3
- [**genai/phi3.5-vision**](./genai/phi3.5-vision/): Multimodal vision-language processing using Phi-3.5-Vision

### GenAI Prerequisites

1. **ONNX Runtime GenAI Library**

   Download from [ONNX Runtime GenAI Releases](https://github.com/microsoft/onnxruntime-genai/releases):
   - macOS: `libonnxruntime-genai.dylib`
   - Linux: `libonnxruntime-genai.so`
   - Windows: `onnxruntime-genai.dll`

2. **Environment Variable**

   Set the library path before running GenAI examples:
   ```bash
   export ONNXRUNTIME_GENAI_LIB_PATH=/path/to/libonnxruntime-genai.dylib
   ```

### GenAI Troubleshooting

1. **Library not found**: Ensure `ONNXRUNTIME_GENAI_LIB_PATH` is set correctly and the library is accessible.

2. **macOS ARM64 crash on library load**: See [onnxruntime-genai issues](https://github.com/microsoft/onnxruntime-genai/issues).
