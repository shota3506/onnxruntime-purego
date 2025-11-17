package onnxruntime

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/shota3506/onnxruntime-purego/internal/cstrings"
)

// apiBase is the common structure across all API versions.
type apiBase struct {
	GetAPI           uintptr // func(version uint32) uintptr
	GetVersionString uintptr // func() *byte
}

// getVersionString returns the ONNX Runtime library version string.
// This is useful for logging and debugging purposes.
func getVersionString(libraryHandle uintptr) (string, error) {
	// Get OrtGetAPIBase function
	var ortGetAPIBase func() *apiBase
	purego.RegisterLibFunc(&ortGetAPIBase, libraryHandle, "OrtGetApiBase")

	// Get the API base
	base := ortGetAPIBase()
	if base == nil {
		return "", fmt.Errorf("OrtGetApiBase returned nil")
	}

	// Register GetVersionString function
	var getVersionStringFunc func() *byte
	purego.RegisterFunc(&getVersionStringFunc, base.GetVersionString)

	// Get version string
	versionPtr := getVersionStringFunc()
	if versionPtr == nil {
		return "", fmt.Errorf("GetVersionString returned nil")
	}

	// SAFETY: This conversion is safe because:
	// 1. The pointer is returned from GetVersionString (C function)
	// 2. It points to a static string in the ONNX Runtime library
	// 3. This string remains valid for the lifetime of the loaded library
	// Safe FFI pattern: pointer to static C string
	return cstrings.CStringToString((*byte)(unsafe.Pointer(versionPtr))), nil
}
