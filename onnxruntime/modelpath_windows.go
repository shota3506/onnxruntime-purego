//go:build windows

package onnxruntime

import (
	"syscall"
	"unsafe"
)

// modelPathArg encodes a model path as ONNX Runtime expects it on this
// platform, returning the backing allocation so the caller can keep it alive
// across the C call. On Windows ORTCHAR_T is wchar_t — UTF-16, not UTF-8.
// Passing a UTF-8 byte string makes ONNX Runtime read the bytes two at a time
// as wide characters, and the load fails with a "file doesn't exist" error
// naming a path of CJK-looking mojibake (the UTF-8 misread as UTF-16). The
// pointer is typed *byte only to match the registered function signature; at
// the ABI level it is just an address.
func modelPathArg(path string) (*byte, any, error) {
	w, err := syscall.UTF16FromString(path)
	if err != nil {
		return nil, nil, err
	}
	return (*byte)(unsafe.Pointer(&w[0])), w, nil
}
