//go:build !windows

package onnxruntime

// modelPathArg encodes a model path as ONNX Runtime expects it on this
// platform, returning the backing allocation so the caller can keep it alive
// across the C call. The session API takes `const ORTCHAR_T *`, and ORTCHAR_T
// is char everywhere except Windows, so on Unix this is the plain
// NUL-terminated byte string.
func modelPathArg(path string) (*byte, any, error) {
	b := append([]byte(path), 0)
	return &b[0], b, nil
}
