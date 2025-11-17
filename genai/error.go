package genai

import (
	"fmt"

	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
	"github.com/shota3506/onnxruntime-purego/internal/cstrings"
)

// resultError checks if an OgaResult indicates an error and returns a formatted error.
func resultError(funcs *api.Funcs, result api.OgaResult) error {
	if result == 0 {
		return nil
	}

	msgPtr := funcs.ResultGetError(result)
	if msgPtr == nil {
		funcs.DestroyResult(result)
		return fmt.Errorf("onnxruntime genai error")
	}

	// Convert C string to Go string
	msg := cstrings.CStringToString(msgPtr)
	funcs.DestroyResult(result)

	return fmt.Errorf("onnxruntime genai: %s", msg)
}
