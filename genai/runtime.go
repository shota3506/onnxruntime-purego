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
//
// Note: This does NOT call OgaShutdown() because it would break other Runtime instances
// in the same process. The ONNX Runtime GenAI C API has the following design limitation:
//
//   - OgaShutdown() destroys global state (KV cache pools, thread pools, etc.) that is
//     shared across all Runtime instances in the process
//   - There is no OgaInitialize() function to reinitialize the library after shutdown
//   - According to the C API documentation, OgaShutdown() is intended to be called only
//     at process exit, not when releasing individual Runtime instances
//
// As a result, calling OgaShutdown() in Close() would prevent creating new Runtime
// instances after closing a previous one, causing segmentation faults when the new
// Runtime tries to access the already-destroyed global state.
//
// The library resources will be automatically cleaned up by the OS when the process exits.
// If you need explicit cleanup at process exit, you can manually call runtime.funcs.Shutdown()
// as the very last operation before your program terminates.
func (r *Runtime) Close() error {
	if r.funcs != nil {
		// Do NOT call r.funcs.Shutdown() - see function documentation above
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
