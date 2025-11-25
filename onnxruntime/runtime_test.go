package onnxruntime

import (
	"slices"
	"testing"
)

func TestGetAvailableProviders(t *testing.T) {
	runtime := newTestRuntime(t)

	providers, err := runtime.GetAvailableProviders()
	if err != nil {
		t.Fatalf("Failed to get available providers: %v", err)
	}

	t.Logf("Available providers: %v", providers)

	if !slices.Contains(providers, "CPUExecutionProvider") {
		t.Error("Expected CPUExecutionProvider to be available")
	}
}
