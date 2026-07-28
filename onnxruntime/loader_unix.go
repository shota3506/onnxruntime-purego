//go:build !windows

package onnxruntime

import "github.com/ebitengine/purego"

// loadLibrary opens a shared library and returns its handle. On Unix this is
// dlopen; the Windows implementation lives in loader_windows.go, since purego
// defines Dlopen only on Unix platforms.
func loadLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
