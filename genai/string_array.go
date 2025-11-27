package genai

import (
	"fmt"

	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
)

// stringArray wraps OgaStringArray for managing arrays of strings.
type stringArray struct {
	ptr     api.OgaStringArray
	runtime *Runtime
	// Keep references to C strings to prevent garbage collection
	cstrings [][]byte
}

// newStringArray creates a new string array from a Go string slice.
func (r *Runtime) newStringArray(strings []string) (*stringArray, error) {
	if len(strings) == 0 {
		return nil, fmt.Errorf("cannot create empty string array")
	}

	// Convert Go strings to null-terminated C strings
	cstrings := make([][]byte, len(strings))
	cstringPtrs := make([]*byte, len(strings))
	for i, s := range strings {
		cstrings[i] = stringToBytes(s)
		cstringPtrs[i] = &cstrings[i][0]
	}

	// Create OgaStringArray from C strings
	var arrayPtr api.OgaStringArray
	result := r.funcs.CreateStringArrayFromStrings(
		&cstringPtrs[0],
		uintptr(len(strings)),
		&arrayPtr,
	)
	if err := resultError(r.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create string array: %w", err)
	}

	return &stringArray{
		ptr:      arrayPtr,
		runtime:  r,
		cstrings: cstrings,
	}, nil
}

// Close releases resources associated with the string array.
func (s *stringArray) Close() {
	if s.ptr != 0 {
		s.runtime.funcs.DestroyStringArray(s.ptr)
		s.ptr = 0
	}
	s.cstrings = nil
}
