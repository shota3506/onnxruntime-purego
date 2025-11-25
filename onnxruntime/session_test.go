package onnxruntime

import (
	"bytes"
	"errors"
	"os"
	"slices"
	"testing"
)

func TestNewSessionFromReader(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}

	session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	if session.ptr == 0 {
		t.Fatal("Session pointer should not be 0")
	}
}

func TestNewSessionWithOptions(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}
	defer env.Close()

	opts := &SessionOptions{
		IntraOpNumThreads: 2,
	}

	modelFile, err := os.Open(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, opts)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()
}

func TestNewSessionFromReaderWithInvalidModel(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}
	defer env.Close()

	invalidModelData := []byte{0x00, 0x01, 0x02, 0x03}
	_, err = runtime.NewSessionFromReader(env, bytes.NewReader(invalidModelData), nil)
	if err == nil {
		t.Fatal("Expected error when loading invalid model data")
	}
}

func TestSessionInputNames(t *testing.T) {
	session := newTestSession(t, newTestRuntime(t))

	inputNames := session.InputNames()
	if !slices.Equal(inputNames, []string{"input"}) {
		t.Errorf("Expected input names ['input'], got %v", inputNames)
	}
}

func TestSessionOutputNames(t *testing.T) {
	session := newTestSession(t, newTestRuntime(t))

	outputNames := session.OutputNames()
	if !slices.Equal(outputNames, []string{"logits"}) {
		t.Errorf("Expected output names ['logits'], got %v", outputNames)
	}
}

func TestSessionRun(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	inputTensor, err := NewTensorValue(runtime, inputData, []int64{1, 10})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	outputs, err := session.Run(t.Context(), map[string]*Value{
		"input": inputTensor,
	})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	defer func() {
		for _, output := range outputs {
			output.Close()
		}
	}()

	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output, got %d", len(outputs))
	}

	// Verify output is a valid tensor
	for name, value := range outputs {
		if value == nil {
			t.Errorf("Output %s is nil", name)
		}

		valueType, err := value.GetValueType()
		if err != nil {
			t.Errorf("Failed to get value type for output %s: %v", name, err)
		}
		if valueType != ONNXTypeTensor {
			t.Errorf("Output %s is not a tensor, got type %d", name, valueType)
		}
	}

	// Verify output data
	outputData, _, err := GetTensorData[float32](outputs["logits"])
	if err != nil {
		t.Fatalf("Failed to get output data: %v", err)
	}

	if len(outputData) != 3 {
		t.Fatalf("Output length mismatch: expected 3, got %d", len(outputData))
	}
}

func TestSessionRunBatch(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{
		1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
		11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
	}
	inputShape := []int64{2, 10}

	inputTensor, err := NewTensorValue(runtime, inputData, inputShape)
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	outputs, err := session.Run(t.Context(), map[string]*Value{
		"input": inputTensor,
	})
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}
	defer func() {
		for _, output := range outputs {
			output.Close()
		}
	}()

	outputData, _, err := GetTensorData[float32](outputs["logits"])
	if err != nil {
		t.Fatalf("Failed to get output data: %v", err)
	}

	expectedOutputLength := 6 // batch_size=2, output_dim=3
	if len(outputData) != expectedOutputLength {
		t.Fatalf("Output length mismatch: expected %d, got %d", expectedOutputLength, len(outputData))
	}
}

func TestSessionRunWithClosedSession(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1.0, 2.0, 3.0}
	inputTensor, err := NewTensorValue(runtime, inputData, []int64{1, 3})
	if err != nil {
		t.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	session.Close()

	_, err = session.Run(t.Context(), map[string]*Value{
		"input": inputTensor,
	})
	if !errors.Is(err, ErrSessionClosed) {
		t.Errorf("Expected ErrSessionClosed, got: %v", err)
	}
}

func TestSessionRunWithWrongInputShape(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	wrongData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	wrongShape := []int64{1, 5}

	tensor, err := NewTensorValue(runtime, wrongData, wrongShape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	_, err = session.Run(t.Context(), map[string]*Value{
		"input": tensor,
	})

	var ortErr *RuntimeError
	if !errors.As(err, &ortErr) {
		t.Errorf("Expected OrtError, got: %v", err)
	}
	if ortErr.Code != ErrorCodeInvalidArgument {
		t.Errorf("Expected OrtInvalidArgument error code, got: %d", ortErr.Code)
	}
}

func TestSessionRunWithWrongInputCount(t *testing.T) {
	runtime := newTestRuntime(t)
	session := newTestSession(t, runtime)

	inputData := []float32{1.0, 2.0, 3.0}
	inputShape := []int64{1, 3}

	tensor, err := NewTensorValue(runtime, inputData, inputShape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	_, err = session.Run(t.Context(), map[string]*Value{
		"input1": tensor,
		"input2": tensor,
	})
	var ortErr *RuntimeError
	if !errors.As(err, &ortErr) {
		t.Errorf("Expected OrtError, got: %v", err)
	}
	if ortErr.Code != ErrorCodeInvalidArgument {
		t.Errorf("Expected OrtInvalidArgument error code, got: %d", ortErr.Code)
	}
}
