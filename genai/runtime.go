package genai

import (
	"fmt"

	"github.com/ebitengine/purego"
	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
)

// Runtime represents an instance of the ONNX Runtime GenAI library.
type Runtime struct {
	libraryHandle uintptr
	funcs         *api.Funcs
}

// NewRuntime loads the ONNX Runtime GenAI shared library from the specified path.
// The libraryPath should point to the GenAI shared library
// (e.g., "libonnxruntime-genai.dylib" on macOS, "libonnxruntime-genai.so" on Linux).
func NewRuntime(libraryPath string) (*Runtime, error) {
	libraryHandle, err := purego.Dlopen(libraryPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("failed to load GenAI library: %w", err)
	}

	funcs, err := api.InitializeFuncs(libraryHandle)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize GenAI API functions: %w", err)
	}

	return &Runtime{
		libraryHandle: libraryHandle,
		funcs:         funcs,
	}, nil
}

// Close releases resources associated with the GenAI library.
// This calls OgaShutdown to clean up global state.
func (r *Runtime) Close() error {
	if r.funcs != nil {
		r.funcs.Shutdown()
		r.funcs = nil
	}

	if r.libraryHandle != 0 {
		r.libraryHandle = 0
	}

	return nil
}

// NewModel loads a model from the specified directory path.
// The path should point to a directory containing the model files
// (e.g., genai_config.json, model.onnx, tokenizer files).
func (r *Runtime) NewModel(modelPath string) (*Model, error) {
	pathBytes := stringToBytes(modelPath)

	var modelPtr api.OgaModel
	result := r.funcs.CreateModel(&pathBytes[0], &modelPtr)
	if err := resultError(r.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create model: %w", err)
	}

	return &Model{
		ptr:     modelPtr,
		runtime: r,
	}, nil
}

// stringToBytes converts a Go string to a null-terminated byte slice.
func stringToBytes(s string) []byte {
	return append([]byte(s), 0)
}
