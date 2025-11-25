package onnxruntime

import (
	"bytes"
	"os"
	"testing"
)

func BenchmarkSessionRun(b *testing.B) {
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		b.Fatalf("Failed to create runtime: %v", err)
	}
	defer runtime.Close()

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		b.Fatalf("Failed to create environment: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		b.Fatalf("Failed to read model file: %v", err)
	}

	session, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
	if err != nil {
		b.Fatalf("Failed to create session: %v", err)
	}
	defer session.Close()

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	inputShape := []int64{1, 10}

	inputTensor, err := NewTensorValue(runtime, inputData, inputShape)
	if err != nil {
		b.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Close()

	inputs := map[string]*Value{
		"input": inputTensor,
	}

	for b.Loop() {
		outputs, err := session.Run(b.Context(), inputs)
		if err != nil {
			b.Fatalf("Failed to run inference: %v", err)
		}
		for _, output := range outputs {
			output.Close()
		}
	}
}
