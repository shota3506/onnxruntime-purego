package onnxruntime

import (
	"log"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

var libraryPath string

func testModelPath() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		log.Fatal("Failed to get current file path")
	}

	// Get the project root directory (parent of onnxruntime package)
	projectRoot := filepath.Dir(filepath.Dir(filename))
	modelPath := filepath.Join(projectRoot, "onnxruntime", "internal", "tests", "testdata", "model.onnx")

	return modelPath
}

func newTestRuntime(t *testing.T) *Runtime {
	t.Helper()

	// Use environment variable if set, otherwise let the system search standard paths
	runtime, err := NewRuntime(libraryPath, 23)
	if err != nil {
		t.Fatalf("Failed to create runtime: %v", err)
	}
	t.Cleanup(func() { runtime.Close() })

	return runtime
}

func newTestSession(t *testing.T, runtime *Runtime) *Session {
	t.Helper()

	env, err := runtime.NewEnv("test", LoggingLevelWarning)
	if err != nil {
		t.Fatalf("Failed to create environment: %v", err)
	}
	t.Cleanup(func() { env.Close() })

	modelFile, err := os.Open(testModelPath())
	if err != nil {
		t.Fatalf("Failed to read model file: %v", err)
	}
	defer modelFile.Close()

	session, err := runtime.NewSessionFromReader(env, modelFile, nil)
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}
	t.Cleanup(func() { session.Close() })

	return session
}

func TestMain(m *testing.M) {
	libraryPath = os.Getenv("ONNXRUNTIME_LIB_PATH")

	m.Run()
}
