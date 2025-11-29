package genai

import (
	"os"
	"testing"
)

var (
	libraryPath   string
	testModelPath string
)

// isModelAvailable checks if a test model path is configured.
func isModelAvailable() bool {
	return testModelPath != ""
}

// newTestRuntime creates a new Runtime for testing.
func newTestRuntime(t *testing.T) *Runtime {
	t.Helper()

	// Use environment variable if set, otherwise let the system search standard paths
	rt, err := NewRuntime(libraryPath)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}
	t.Cleanup(func() { rt.Close() })

	return rt
}

// newTestModel creates a new Model for testing.
func newTestModel(t *testing.T, rt *Runtime) *Model {
	t.Helper()

	if !isModelAvailable() {
		t.Skip("Test model not available. Set ONNXRUNTIME_GENAI_MODEL_PATH environment variable.")
	}

	model, err := rt.NewModel(testModelPath, nil)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	t.Cleanup(func() { model.Close() })

	return model
}

// newTestTokenizer creates a new Tokenizer for testing.
func newTestTokenizer(t *testing.T, model *Model) *Tokenizer {
	t.Helper()

	tokenizer, err := model.NewTokenizer()
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}
	t.Cleanup(func() { tokenizer.Close() })

	return tokenizer
}

func TestMain(m *testing.M) {
	libraryPath = os.Getenv("ONNXRUNTIME_GENAI_LIB_PATH")
	testModelPath = os.Getenv("ONNXRUNTIME_GENAI_MODEL_PATH")

	m.Run()
}
