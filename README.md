# onnxruntime-purego
[![Go Reference](https://pkg.go.dev/badge/github.com/shota3506/onnxruntime-purego.svg)](https://pkg.go.dev/github.com/shota3506/onnxruntime-purego)

Pure Go bindings for [ONNX Runtime](https://github.com/microsoft/onnxruntime) using [ebitengine/purego](https://github.com/ebitengine/purego).

This library provides a pure Go interface to ONNX Runtime without requiring cgo, enabling cross-platform machine learning inference in Go applications.

NOTE: This project is currently unstable. APIs may change without notice.

## Supported Versions

| Library | Supported Version |
|---------|-------------------|
| ONNX Runtime | 1.23.x |
| ONNX Runtime GenAI | 0.11.x |

## ONNX Runtime GenAI Support

This library also includes experimental support for [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), enabling text generation with large language models. See [`examples/`](./examples/) for usage examples.

## Prerequisites

You need to have the ONNX Runtime shared library installed on your system:

- **macOS**: `libonnxruntime.dylib`
- **Linux**: `libonnxruntime.so`
- **Windows**: `onnxruntime.dll`

Download the appropriate library from the [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

The library will be automatically discovered if placed in standard system locations:

- **macOS**: `/usr/local/lib`, `/opt/homebrew/lib`, `/usr/lib`
- **Linux**: `/usr/local/lib`, `/usr/lib`, `/lib`
- **Windows**: Standard DLL search paths

Alternatively, you can specify a custom path when creating the runtime.

## Installation

```bash
go get github.com/shota3506/onnxruntime-purego
```

## Examples

See the [`examples/`](./examples/) directory for complete usage examples.
