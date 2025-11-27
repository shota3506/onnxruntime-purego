package onnxruntime

import (
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/shota3506/onnxruntime-purego/internal/cstrings"
	"github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api"
	v23 "github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api/v23"
)

// supportedAPIVersions lists all API versions supported by this library.
var supportedAPIVersions = []uint32{23}

// getDefaultLibraryName returns the default library name based on the current platform.
func getDefaultLibraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "linux":
		return "libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so"
	}
}

// Runtime represents an instance of the ONNX Runtime library.
// Multiple Runtime instances can coexist, allowing the use of different
// ONNX Runtime versions simultaneously.
type Runtime struct {
	libraryHandle uintptr
	apiVersion    uint32
	versionString string

	// API function pointers (version-specific)
	apiFuncs api.APIFuncs

	// Default allocator and memory info
	allocator     *allocator
	cpuMemoryInfo *memoryInfo
}

// NewRuntime loads the ONNX Runtime shared library from the specified path and
// initializes the C API interface with the specified API version.
// The libraryPath should point to the ONNX Runtime shared library
// (e.g., "libonnxruntime.so", "libonnxruntime.dylib", or "onnxruntime.dll").
// If libraryPath is empty, the system will search for the library in standard locations
// The apiVersion parameter specifies which ONNX Runtime C API version to use (e.g., 23, 24).
func NewRuntime(libraryPath string, apiVersion uint32) (*Runtime, error) {
	// Validate API version is supported
	if !isSupportedAPIVersion(apiVersion) {
		return nil, fmt.Errorf("unsupported API version %d (supported: %v)", apiVersion, supportedAPIVersions)
	}

	// If no path is provided, use the default library name and let the system search standard paths
	if libraryPath == "" {
		libraryPath = getDefaultLibraryName()
	}

	libraryHandle, err := purego.Dlopen(libraryPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("failed to load library: %w", err)
	}

	versionString, err := getVersionString(libraryHandle)
	if err != nil {
		// Non-fatal, just use a default
		versionString = fmt.Sprintf("unknown (API version %d)", apiVersion)
	}

	runtime := &Runtime{
		libraryHandle: libraryHandle,
		apiVersion:    apiVersion,
		versionString: versionString,
	}

	// Initialize API functions based on specified version
	if err := runtime.initializeAPI(); err != nil {
		return nil, fmt.Errorf("failed to initialize API: %w", err)
	}

	// Initialize default allocator
	if err := runtime.initializeAllocator(); err != nil {
		return nil, fmt.Errorf("failed to initialize allocator: %w", err)
	}

	// Initialize default CPU memory info
	if err := runtime.initializeMemoryInfo(); err != nil {
		return nil, fmt.Errorf("failed to initialize memory info: %w", err)
	}

	return runtime, nil
}

// isSupportedAPIVersion checks if the given API version is supported.
func isSupportedAPIVersion(version uint32) bool {
	return slices.Contains(supportedAPIVersions, version)
}

// initializeAPI initializes the API function pointers based on the detected version.
func (r *Runtime) initializeAPI() error {
	switch r.apiVersion {
	case 23:
		return r.initializeV23API()
	default:
		return fmt.Errorf("unsupported API version: %d", r.apiVersion)
	}
}

// initializeV23API initializes the v23 API function pointers.
func (r *Runtime) initializeV23API() error {
	apiFuncs, err := v23.InitializeFuncs(r.libraryHandle)
	if err != nil {
		return err
	}
	r.apiFuncs = apiFuncs
	return nil
}

// initializeAllocator initializes the default allocator for this runtime.
func (r *Runtime) initializeAllocator() error {
	var allocPtr api.OrtAllocator
	status := r.apiFuncs.GetAllocatorWithDefaultOptions(&allocPtr)
	if err := r.statusError(status); err != nil {
		return fmt.Errorf("failed to get default allocator: %w", err)
	}

	r.allocator = &allocator{
		ptr:     allocPtr,
		runtime: r,
	}
	return nil
}

// initializeMemoryInfo initializes the default CPU memory info for this runtime.
func (r *Runtime) initializeMemoryInfo() error {
	memInfo, err := r.createCPUMemoryInfo(allocatorTypeDevice, memTypeCPU)
	if err != nil {
		return fmt.Errorf("failed to create CPU memory info: %w", err)
	}
	r.cpuMemoryInfo = memInfo
	return nil
}

// createCPUMemoryInfo creates memory info for CPU.
func (r *Runtime) createCPUMemoryInfo(allocType allocatorType, memType memType) (*memoryInfo, error) {
	var memInfoPtr api.OrtMemoryInfo
	status := r.apiFuncs.CreateCpuMemoryInfo(allocType, memType, &memInfoPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create CPU memory info: %w", err)
	}

	return &memoryInfo{
		ptr:     memInfoPtr,
		runtime: r,
	}, nil
}

// Close releases resources associated with the ONNX Runtime library.
// This should be called when the runtime is no longer needed, typically
// using defer after NewRuntime. It is safe to call Close multiple times.
func (r *Runtime) Close() error {
	// Release default memory info
	if r.cpuMemoryInfo != nil {
		r.cpuMemoryInfo.release()
		r.cpuMemoryInfo = nil
	}

	// Clear default allocator (no release needed - managed by ONNX Runtime)
	r.allocator = nil

	// Clear cached function pointers
	r.apiFuncs = nil

	if r.libraryHandle != 0 {
		// Note: purego doesn't provide Dlclose, so we just clear our reference
		r.libraryHandle = 0
	}
	return nil
}

// GetAPIVersion returns the API version of this runtime instance.
func (r *Runtime) GetAPIVersion() uint32 {
	return r.apiVersion
}

// GetVersionString returns the version string of the ONNX Runtime library.
func (r *Runtime) GetVersionString() string {
	return r.versionString
}

// statusError converts an OrtStatus to a Go error.
func (r *Runtime) statusError(status api.OrtStatus) error {
	if status == 0 {
		return nil
	}

	code := r.apiFuncs.GetErrorCode(status)
	messagePtr := r.apiFuncs.GetErrorMessage(status)

	message := cstrings.CStringToString((*byte)(messagePtr))

	r.apiFuncs.ReleaseStatus(status)

	return &RuntimeError{
		Code:    code,
		Message: message,
	}
}

// GetAvailableProviders returns a list of available execution provider names.
// The list includes all execution providers that are compiled into the ONNX Runtime library.
// Common providers include "CPUExecutionProvider", "CUDAExecutionProvider", etc.
func (r *Runtime) GetAvailableProviders() ([]string, error) {
	var providersPtr **byte
	var length int32
	status := r.apiFuncs.GetAvailableProviders(&providersPtr, &length)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get available providers: %w", err)
	}

	providers := make([]string, length)
	if length > 0 {
		providerPtrs := unsafe.Slice(providersPtr, length)
		for i := int32(0); i < length; i++ {
			providers[i] = cstrings.CStringToString(providerPtrs[i])
		}
	}

	// Release the allocated provider list
	status = r.apiFuncs.ReleaseAvailableProviders(providersPtr, length)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to release available providers: %w", err)
	}

	return providers, nil
}
