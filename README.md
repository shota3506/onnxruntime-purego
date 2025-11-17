# onnxruntime-purego

Pure Go bindings for [ONNX Runtime](https://github.com/microsoft/onnxruntime) using [ebitengine/purego](https://github.com/ebitengine/purego).

This library provides a pure Go interface to ONNX Runtime without requiring cgo, enabling cross-platform machine learning inference in Go applications.

NOTE: This project is currently unstable. APIs may change without notice.

## ONNX Runtime GenAI Support (Experimental)

This library also includes experimental support for [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), enabling text generation with large language models. See `examples/genai/` for usage examples.

## Prerequisites

NOTE: You need to have the ONNX Runtime shared library installed on your system:

- **macOS**: `libonnxruntime.dylib`
- **Linux**: `libonnxruntime.so`
- **Windows**: `onnxruntime.dll`

Download the appropriate library from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

## Installation

```bash
go get github.com/shota3506/onnxruntime-purego
```

## Examples

The `examples/` directory contains complete examples:

- **resnet**: Image classification using ResNet
- **roberta-sentiment**: Sentiment analysis using RoBERTa
- **yolov10**: Object detection using YOLOv10
- **genai**: Text generation using ONNX Runtime GenAI (experimental)

See each example's README for detailed instructions.
