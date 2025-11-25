package onnxruntime

import (
	"errors"
	"strings"
	"testing"
)

func TestStatusError(t *testing.T) {
	runtime := newTestRuntime(t)

	t.Run("nil status returns nil error", func(t *testing.T) {
		err := runtime.statusError(0)
		if err != nil {
			t.Errorf("runtime.statusError(0) should return nil, got %v", err)
		}
	})

	t.Run("error codes", func(t *testing.T) {
		testCases := []struct {
			code ErrorCode
			name string
		}{
			{ErrorCodeFail, "OrtFail"},
			{ErrorCodeInvalidArgument, "OrtInvalidArgument"},
			{ErrorCodeNoSuchFile, "OrtNoSuchFile"},
			{ErrorCodeNoModel, "OrtNoModel"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				msgBytes := append([]byte("error for "+tc.name), 0)
				status := runtime.apiFuncs.CreateStatus(tc.code, &msgBytes[0])
				if status == 0 {
					t.Fatal("createStatus should return non-zero status")
				}

				err := runtime.statusError(status)
				if err == nil {
					t.Fatal("statusError should return an error")
				}

				var ortErr *RuntimeError
				if !errors.As(err, &ortErr) {
					t.Fatalf("error should be of type *Error")
				}

				if ortErr.Code != tc.code {
					t.Errorf("Expected error code %d, got %d", tc.code, ortErr.Code)
				}

				if !strings.Contains(err.Error(), "error for "+tc.name) {
					t.Errorf("Error message should contain 'error for %s', got: %s", tc.name, err.Error())
				}
			})
		}
	})
}

func TestStatusFunctions(t *testing.T) {
	runtime := newTestRuntime(t)

	// Test creating a status
	msgBytes := append([]byte("test error"), 0)
	status := runtime.apiFuncs.CreateStatus(ErrorCodeFail, &msgBytes[0])
	if status == 0 {
		t.Fatal("createStatus should return non-zero status")
	}
	defer runtime.apiFuncs.ReleaseStatus(status)

	// Test getting error code
	code := runtime.apiFuncs.GetErrorCode(status)
	if code != ErrorCodeFail {
		t.Errorf("Expected error code %d, got %d", ErrorCodeFail, code)
	}

	// Test getting error message
	msgPtrVal := runtime.apiFuncs.GetErrorMessage(status)
	if msgPtrVal == 0 {
		t.Error("Error message pointer should not be 0")
	}
}

func TestReleaseStatusWithNullStatus(t *testing.T) {
	runtime := newTestRuntime(t)

	// This should not crash
	runtime.apiFuncs.ReleaseStatus(0)
}
