package genai

import (
	"testing"
)

func TestNewModel(t *testing.T) {
	if !isModelAvailable() {
		t.Skip("Test model not available. Set ONNXRUNTIME_GENAI_MODEL_PATH environment variable.")
	}

	rt := newTestRuntime(t)

	model, err := rt.NewModel(testModelPath)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	defer model.Close()

	if model == nil {
		t.Fatal("Model is nil")
	}
	if model.ptr == 0 {
		t.Fatal("Model ptr is null")
	}
	if model.runtime != rt {
		t.Fatal("Model runtime does not match")
	}
}

func TestModelClose(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	// Close should succeed
	model.Close()

	// Second close should also succeed (idempotent)
	model.Close()
}

func TestModelNewTokenizer(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	tokenizer, err := model.NewTokenizer()
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}
	defer tokenizer.Close()

	if tokenizer == nil {
		t.Fatal("Tokenizer is nil")
	}
	if tokenizer.ptr == 0 {
		t.Fatal("Tokenizer ptr is null")
	}
}

func TestModelNewGenerator(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	// Create generator with default params
	generator, err := model.NewGenerator(nil)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}
	defer generator.Close()

	if generator == nil {
		t.Fatal("Generator is nil")
	}
	if generator.ptr == 0 {
		t.Fatal("Generator ptr is null")
	}
}

func TestModelNewGeneratorWithParams(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	params := GeneratorParams{
		"max_length":  int(20),
		"temperature": float64(0.7),
		"do_sample":   true,
	}

	generator, err := model.NewGenerator(params)
	if err != nil {
		t.Fatalf("Failed to create generator with params: %v", err)
	}
	defer generator.Close()

	if generator == nil {
		t.Fatal("Generator is nil")
	}
}
