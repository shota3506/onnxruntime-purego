package cstrings

import "unsafe"

// CStringToString converts a C-style null-terminated string to a Go string.
func CStringToString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(length))) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice(ptr, length))
}
