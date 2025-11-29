package genai

import (
	"fmt"
	"runtime"

	"github.com/ebitengine/purego"
	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
)

// getDefaultLibraryName returns the default GenAI library name based on the current platform.
func getDefaultLibraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime-genai.dylib"
	case "linux":
		return "libonnxruntime-genai.so"
	case "windows":
		return "onnxruntime-genai.dll"
	default:
		return "libonnxruntime-genai.so"
	}
}

// Runtime represents an instance of the ONNX Runtime GenAI library.
type Runtime struct {
	libraryHandle uintptr
	funcs         *api.Funcs
}

// NewRuntime loads the ONNX Runtime GenAI shared library from the specified path.
// The libraryPath should point to the GenAI shared library
// (e.g., "libonnxruntime-genai.dylib" on macOS, "libonnxruntime-genai.so" on Linux).
// If libraryPath is empty, the system will search for the library in standard locations
func NewRuntime(libraryPath string) (*Runtime, error) {
	// If no path is provided, use the default library name and let the system search standard paths
	if libraryPath == "" {
		libraryPath = getDefaultLibraryName()
	}

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

// ProviderOptions contains configuration options for an execution provider.
type ProviderOptions map[string]string

// ModelOptions configures options for creating a model.
type ModelOptions struct {
	// Providers specifies the execution providers to use, in order of preference.
	Providers []string

	// ProviderOptions specifies options for each provider.
	// The key is the provider name, and the value is a map of option key-value pairs.
	ProviderOptions map[string]ProviderOptions
}

// NewModel loads a model from the specified directory path.
// The path should point to a directory containing the model files
// (e.g., genai_config.json, model.onnx, tokenizer files).
// If options is nil, default options will be used.
func (r *Runtime) NewModel(modelPath string, options *ModelOptions) (*Model, error) {
	pathBytes := stringToBytes(modelPath)

	// If no options or no providers specified, use the simple CreateModel
	if options == nil || len(options.Providers) == 0 {
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

	// Create config for provider configuration
	var configPtr api.OgaConfig
	result := r.funcs.CreateConfig(&pathBytes[0], &configPtr)
	if err := resultError(r.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create config: %w", err)
	}
	defer r.funcs.DestroyConfig(configPtr)

	// Clear existing providers
	result = r.funcs.ConfigClearProviders(configPtr)
	if err := resultError(r.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to clear providers: %w", err)
	}

	// Append providers
	for _, provider := range options.Providers {
		providerBytes := stringToBytes(provider)
		result = r.funcs.ConfigAppendProvider(configPtr, &providerBytes[0])
		if err := resultError(r.funcs, result); err != nil {
			return nil, fmt.Errorf("failed to append provider %q: %w", provider, err)
		}

		// Set provider options if specified
		if options.ProviderOptions != nil {
			if providerOpts, ok := options.ProviderOptions[provider]; ok {
				for key, value := range providerOpts {
					keyBytes := stringToBytes(key)
					valueBytes := stringToBytes(value)
					result = r.funcs.ConfigSetProviderOption(configPtr, &providerBytes[0], &keyBytes[0], &valueBytes[0])
					if err := resultError(r.funcs, result); err != nil {
						return nil, fmt.Errorf("failed to set provider option %q=%q for %q: %w", key, value, provider, err)
					}
				}
			}
		}
	}

	var modelPtr api.OgaModel
	result = r.funcs.CreateModelFromConfig(configPtr, &modelPtr)
	if err := resultError(r.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create model from config: %w", err)
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
