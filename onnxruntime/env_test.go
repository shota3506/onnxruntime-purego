package onnxruntime

import (
	"testing"
)

func TestNewEnv(t *testing.T) {
	runtime := newTestRuntime(t)

	logLevels := []LoggingLevel{
		LoggingLevelVerbose,
		LoggingLevelInfo,
		LoggingLevelWarning,
		LoggingLevelError,
		LoggingLevelFatal,
	}

	for _, level := range logLevels {
		env, err := runtime.NewEnv("test", level)
		if err != nil {
			t.Fatalf("Failed to create environment: %v", err)
			continue
		}
		defer env.Close()

		if env.ptr == 0 {
			t.Fatal("Environment pointer should not be 0")
		}
	}
}

func TestEnvClose(t *testing.T) {
	runtime := newTestRuntime(t)

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}

	env.Close()

	if env.ptr != 0 {
		t.Error("Environment pointer should be 0 after Close()")
	}

	// Second close should not panic
	env.Close()
}
