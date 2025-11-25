package genai

import (
	"testing"
)

func TestNewRuntime(t *testing.T) {
	rt, err := NewRuntime(libraryPath)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}
	defer rt.Close()

	if rt == nil {
		t.Fatal("Runtime is nil")
	}
	if rt.funcs == nil {
		t.Fatal("Runtime funcs is nil")
	}
}

func TestRuntimeClose(t *testing.T) {
	rt, err := NewRuntime(libraryPath)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}

	// Close should succeed
	if err := rt.Close(); err != nil {
		t.Fatalf("Failed to close runtime: %v", err)
	}

	// Second close should also succeed (idempotent)
	if err := rt.Close(); err != nil {
		t.Fatalf("Failed to close runtime second time: %v", err)
	}
}
