package tests

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/shota3506/onnxruntime-purego/onnxruntime"
)

var (
	testDataDir string
	testRuntime *onnxruntime.Runtime
)

const (
	defaultAbsoluteTolerance = 1e-4
	defaultRelativeTolerance = 1e-3
)

func TestMain(m *testing.M) {
	_, filename, _, _ := runtime.Caller(0)
	testDataDir = filepath.Join(filepath.Dir(filename), "testdata")

	// Check if test data exists
	if _, err := os.Stat(testDataDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Test data not found at: %s\n", testDataDir)
		fmt.Fprintf(os.Stderr, "Please run: ./download_test_data.sh\n")
		os.Exit(1)
	}

	libPath := os.Getenv("ONNXRUNTIME_LIB_PATH")

	var err error
	testRuntime, err = onnxruntime.NewRuntime(libPath, 23)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create runtime: %v\n", err)
		os.Exit(1)
	}
	defer testRuntime.Close()

	os.Exit(m.Run())
}

// compareFloat32Tensors compares two float32 tensors with tolerance
// This follows microsoft/onnxruntime's comparison logic
func compareFloat32Tensors(t *testing.T, actual []float32, actualShape []int64, expected []float32, expectedShape []int64) {
	t.Helper()

	if !slices.Equal(actualShape, expectedShape) {
		t.Errorf("Shape mismatch: actual %v vs expected %v", actualShape, expectedShape)
		return
	}
	if len(actual) != len(expected) {
		t.Errorf("Length mismatch: actual %d vs expected %d", len(actual), len(expected))
		return
	}

	// Compare values with tolerance
	maxAbsError := float32(0.0)
	maxRelError := float32(0.0)
	errorCount := 0

	for i := range actual {
		absError := float32(math.Abs(float64(actual[i] - expected[i])))
		relError := float32(0.0)
		if expected[i] != 0 {
			relError = absError / float32(math.Abs(float64(expected[i])))
		}

		if absError > maxAbsError {
			maxAbsError = absError
		}
		if relError > maxRelError {
			maxRelError = relError
		}

		if absError > defaultAbsoluteTolerance && relError > defaultRelativeTolerance {
			if errorCount < 10 { // Limit error output
				t.Errorf("Value mismatch at index %d: actual %.6f vs expected %.6f (abs_err=%.2e, rel_err=%.2e)",
					i, actual[i], expected[i], absError, relError)
			}
			errorCount++
		}
	}

	if errorCount > 0 {
		t.Errorf("Total %d values exceeded tolerance (max_abs_err=%.2e, max_rel_err=%.2e)",
			errorCount, maxAbsError, maxRelError)
	}
}

// compareInt64Tensors compares two int64 tensors (exact match)
func compareInt64Tensors(t *testing.T, actual []int64, actualShape []int64, expected []int64, expectedShape []int64) {
	t.Helper()

	if !slices.Equal(actualShape, expectedShape) {
		t.Errorf("Shape mismatch: actual %v vs expected %v", actualShape, expectedShape)
		return
	}
	if len(actual) != len(expected) {
		t.Errorf("Length mismatch: actual %d vs expected %d", len(actual), len(expected))
		return
	}

	// Compare values (exact match for integers)
	errorCount := 0
	for i := range actual {
		if actual[i] != expected[i] {
			if errorCount < 10 {
				t.Errorf("Value mismatch at index %d: actual %d vs expected %d",
					i, actual[i], expected[i])
			}
			errorCount++
		}
	}

	if errorCount > 0 {
		t.Errorf("Total %d values mismatched", errorCount)
	}
}

// compareInt32Tensors compares two int32 tensors (exact match)
func compareInt32Tensors(t *testing.T, actual []int32, actualShape []int64, expected []int32, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareInt16Tensors compares two int16 tensors (exact match)
func compareInt16Tensors(t *testing.T, actual []int16, actualShape []int64, expected []int16, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareInt8Tensors compares two int8 tensors (exact match)
func compareInt8Tensors(t *testing.T, actual []int8, actualShape []int64, expected []int8, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareUint8Tensors compares two uint8 tensors (exact match)
func compareUint8Tensors(t *testing.T, actual []uint8, actualShape []int64, expected []uint8, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareUint16Tensors compares two uint16 tensors (exact match)
func compareUint16Tensors(t *testing.T, actual []uint16, actualShape []int64, expected []uint16, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareUint32Tensors compares two uint32 tensors (exact match)
func compareUint32Tensors(t *testing.T, actual []uint32, actualShape []int64, expected []uint32, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareUint64Tensors compares two uint64 tensors (exact match)
func compareUint64Tensors(t *testing.T, actual []uint64, actualShape []int64, expected []uint64, expectedShape []int64) {
	compareIntegerTensors(t, len(actual), len(expected), actualShape, expectedShape, func(i int) (int64, int64) {
		return int64(actual[i]), int64(expected[i])
	})
}

// compareIntegerTensors is a generic helper for integer comparison
func compareIntegerTensors(t *testing.T, actualLen, expectedLen int, actualShape, expectedShape []int64, getValue func(int) (int64, int64)) {
	t.Helper()

	if !slices.Equal(actualShape, expectedShape) {
		t.Errorf("Shape mismatch: actual %v vs expected %v", actualShape, expectedShape)
		return
	}
	if actualLen != expectedLen {
		t.Errorf("Length mismatch: actual %d vs expected %d", actualLen, expectedLen)
		return
	}

	// Compare values (exact match for integers)
	errorCount := 0
	for i := range actualLen {
		actualVal, expectedVal := getValue(i)
		if actualVal != expectedVal {
			if errorCount < 10 {
				t.Errorf("Value mismatch at index %d: actual %d vs expected %d",
					i, actualVal, expectedVal)
			}
			errorCount++
		}
	}

	if errorCount > 0 {
		t.Errorf("Total %d values mismatched", errorCount)
	}
}
