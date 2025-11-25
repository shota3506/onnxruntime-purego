package onnxruntime

import (
	"slices"
	"testing"
)

// assertTensorData is a helper function to verify tensor data and shape
func assertTensorData[T TensorData](t *testing.T, tensor *Value, expectedData []T, expectedShape []int64) {
	t.Helper()

	// Get tensor data and shape
	data, shape, err := GetTensorData[T](tensor)
	if err != nil {
		t.Fatalf("Failed to get tensor data: %v", err)
	}

	// Verify shape
	if !slices.Equal(shape, expectedShape) {
		t.Errorf("Shape mismatch: expected %v, got %v", expectedShape, shape)
	}

	// Verify data
	if !slices.Equal(data, expectedData) {
		t.Errorf("Data mismatch: expected %v, got %v", expectedData, data)
	}
}

func TestValueGetValueType(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []float32{1.0, 2.0, 3.0}
	shape := []int64{1, 3}

	tensor, err := NewTensorValue(runtime, data, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	valueType, err := tensor.GetValueType()
	if err != nil {
		t.Fatalf("Failed to get value type: %v", err)
	}

	if valueType != ONNXTypeTensor {
		t.Errorf("Expected value type to be ONNXTypeTensor, got %d", valueType)
	}
}

func TestValueGetTensorShape(t *testing.T) {
	runtime := newTestRuntime(t)

	originalShape := []int64{2, 3, 4}
	data := make([]float32, 2*3*4)
	for i := range data {
		data[i] = float32(i)
	}

	tensor, err := NewTensorValue(runtime, data, originalShape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	retrievedShape, err := tensor.GetTensorShape()
	if err != nil {
		t.Fatalf("Failed to get shape: %v", err)
	}

	if !slices.Equal(retrievedShape, originalShape) {
		t.Errorf("Shape mismatch: expected %v, got %v", originalShape, retrievedShape)
	}
}

func TestValueGetTensorElementType(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []float32{1.0, 2.0, 3.0}
	shape := []int64{1, 3}

	tensor, err := NewTensorValue(runtime, data, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	elemType, err := tensor.GetTensorElementType()
	if err != nil {
		t.Fatalf("Failed to get element type: %v", err)
	}

	if elemType != ONNXTensorElementDataTypeFloat {
		t.Errorf("Expected float type, got %d", elemType)
	}
}

func TestValueGetTensorElementCount(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	shape := []int64{2, 3}

	tensor, err := NewTensorValue(runtime, data, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}
	defer tensor.Close()

	count, err := tensor.GetTensorElementCount()
	if err != nil {
		t.Fatalf("Failed to get element count: %v", err)
	}

	expectedCount := 6
	if count != expectedCount {
		t.Errorf("Expected element count %d, got %d", expectedCount, count)
	}
}

func TestValueClose(t *testing.T) {
	runtime := newTestRuntime(t)

	data := []float32{1.0, 2.0, 3.0}
	shape := []int64{1, 3}

	tensor, err := NewTensorValue(runtime, data, shape)
	if err != nil {
		t.Fatalf("Failed to create tensor: %v", err)
	}

	tensor.Close()
	if tensor.ptr != 0 {
		t.Error("Tensor pointer should be 0 after Close()")
	}

	// Second close should not panic
	tensor.Close()
}

func TestNewTensorValue(t *testing.T) {
	runtime := newTestRuntime(t)

	t.Run("Float32", func(t *testing.T) {
		data := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		shape := []int64{2, 3}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create float32 tensor: %v", err)
		}
		defer tensor.Close()

		// Verify element type
		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeFloat {
			t.Errorf("Expected float type, got %d", elemType)
		}

		// Verify shape
		retrievedShape, err := tensor.GetTensorShape()
		if err != nil {
			t.Fatalf("Failed to get shape: %v", err)
		}
		if !slices.Equal(retrievedShape, shape) {
			t.Errorf("Shape mismatch: expected %v, got %v", shape, retrievedShape)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Float64", func(t *testing.T) {
		data := []float64{1.0, 2.0, 3.0, 4.0}
		shape := []int64{2, 2}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create float64 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeDouble {
			t.Errorf("Expected double type, got %d", elemType)
		}

		retrievedShape, err := tensor.GetTensorShape()
		if err != nil {
			t.Fatalf("Failed to get shape: %v", err)
		}
		if !slices.Equal(retrievedShape, shape) {
			t.Errorf("Shape mismatch: expected %v, got %v", shape, retrievedShape)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Int16", func(t *testing.T) {
		data := []int16{1, 2, 3, 4, 5, 6}
		shape := []int64{3, 2}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create int16 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeInt16 {
			t.Errorf("Expected int16 type, got %d", elemType)
		}

		retrievedShape, err := tensor.GetTensorShape()
		if err != nil {
			t.Fatalf("Failed to get shape: %v", err)
		}
		if !slices.Equal(retrievedShape, shape) {
			t.Errorf("Shape mismatch: expected %v, got %v", shape, retrievedShape)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Int32", func(t *testing.T) {
		data := []int32{10, 20, 30, 40}
		shape := []int64{4}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create int32 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeInt32 {
			t.Errorf("Expected int32 type, got %d", elemType)
		}

		retrievedShape, err := tensor.GetTensorShape()
		if err != nil {
			t.Fatalf("Failed to get shape: %v", err)
		}
		if !slices.Equal(retrievedShape, shape) {
			t.Errorf("Shape mismatch: expected %v, got %v", shape, retrievedShape)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Int64", func(t *testing.T) {
		data := []int64{100, 200, 300}
		shape := []int64{3, 1}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create int64 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeInt64 {
			t.Errorf("Expected int64 type, got %d", elemType)
		}

		retrievedShape, err := tensor.GetTensorShape()
		if err != nil {
			t.Fatalf("Failed to get shape: %v", err)
		}
		if !slices.Equal(retrievedShape, shape) {
			t.Errorf("Shape mismatch: expected %v, got %v", shape, retrievedShape)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Int8", func(t *testing.T) {
		data := []int8{-10, 0, 10, 20}
		shape := []int64{4}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create int8 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeInt8 {
			t.Errorf("Expected int8 type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Uint8", func(t *testing.T) {
		data := []uint8{0, 10, 20, 255}
		shape := []int64{2, 2}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create uint8 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeUint8 {
			t.Errorf("Expected uint8 type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Uint16", func(t *testing.T) {
		data := []uint16{100, 200, 300}
		shape := []int64{3}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create uint16 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeUint16 {
			t.Errorf("Expected uint16 type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Uint32", func(t *testing.T) {
		data := []uint32{1000, 2000, 3000}
		shape := []int64{3}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create uint32 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeUint32 {
			t.Errorf("Expected uint32 type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Uint64", func(t *testing.T) {
		data := []uint64{10000, 20000, 30000}
		shape := []int64{3}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create uint64 tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeUint64 {
			t.Errorf("Expected uint64 type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("Bool", func(t *testing.T) {
		data := []bool{true, false, true, false}
		shape := []int64{4}

		tensor, err := NewTensorValue(runtime, data, shape)
		if err != nil {
			t.Fatalf("Failed to create bool tensor: %v", err)
		}
		defer tensor.Close()

		elemType, err := tensor.GetTensorElementType()
		if err != nil {
			t.Fatalf("Failed to get element type: %v", err)
		}
		if elemType != ONNXTensorElementDataTypeBool {
			t.Errorf("Expected bool type, got %d", elemType)
		}

		// Verify data
		assertTensorData(t, tensor, data, shape)
	})

	t.Run("EmptyData", func(t *testing.T) {
		// Test with empty float32 slice
		_, err := NewTensorValue(runtime, []float32{}, []int64{0})
		if err == nil {
			t.Error("Expected error when creating tensor with empty float32 data")
		}

		// Test with empty int64 slice
		_, err = NewTensorValue(runtime, []int64{}, []int64{0})
		if err == nil {
			t.Error("Expected error when creating tensor with empty int64 data")
		}
	})
}

func TestGetTensorData(t *testing.T) {
	runtime := newTestRuntime(t)

	t.Run("Float32", func(t *testing.T) {
		originalData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
		originalShape := []int64{2, 3}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Float64", func(t *testing.T) {
		originalData := []float64{1.5, 2.5, 3.5, 4.5}
		originalShape := []int64{2, 2}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Int16", func(t *testing.T) {
		originalData := []int16{1, 2, 3, 4, 5, 6, 7, 8}
		originalShape := []int64{2, 2, 2}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Int32", func(t *testing.T) {
		originalData := []int32{10, 20, 30, 40, 50}
		originalShape := []int64{5}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Int64", func(t *testing.T) {
		originalData := []int64{100, 200, 300, 400, 500, 600}
		originalShape := []int64{3, 2}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Uint8", func(t *testing.T) {
		originalData := []uint8{0, 10, 20, 30, 255}
		originalShape := []int64{5}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("Bool", func(t *testing.T) {
		originalData := []bool{true, false, true, false, true, false}
		originalShape := []int64{2, 3}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})

	t.Run("TypeMismatch", func(t *testing.T) {
		// Create a float32 tensor
		originalData := []float32{1.0, 2.0, 3.0}
		originalShape := []int64{3}

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		// Try to extract as int32 (should fail)
		_, _, err = GetTensorData[int32](tensor)
		if err == nil {
			t.Error("Expected error when extracting float32 tensor as int32")
		}
	})

	t.Run("MultiDimensional", func(t *testing.T) {
		// Create a 3D tensor
		originalData := []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
			17, 18, 19, 20,
			21, 22, 23, 24,
		}
		originalShape := []int64{2, 3, 4} // 2x3x4 tensor

		tensor, err := NewTensorValue(runtime, originalData, originalShape)
		if err != nil {
			t.Fatalf("Failed to create tensor: %v", err)
		}
		defer tensor.Close()

		assertTensorData(t, tensor, originalData, originalShape)
	})
}
