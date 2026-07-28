//go:build windows

package genai

import "syscall"

// loadLibrary opens a DLL and returns its handle. purego has no Dlopen on
// Windows; libraries are opened through the Win32 loader, and the resulting
// HMODULE is the same uintptr handle purego's RegisterLibFunc consumes. A bare
// file name is resolved by the standard Windows search order, matching the
// default-library behavior on the other platforms.
func loadLibrary(path string) (uintptr, error) {
	handle, err := syscall.LoadLibrary(path)
	if err != nil {
		return 0, err
	}
	return uintptr(handle), nil
}
