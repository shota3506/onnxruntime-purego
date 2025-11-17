// Package genai provides Go bindings for ONNX Runtime GenAI, enabling efficient
// execution of large language models (LLMs) without CGo.
//
// This package uses purego to call the ONNX Runtime GenAI C API directly,
// providing automatic tokenization, generation loops, and KV cache management
// for transformer-based generative models.
package genai
