//go:build !windows

package onnxruntime

import "github.com/ebitengine/purego"

func loadLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
