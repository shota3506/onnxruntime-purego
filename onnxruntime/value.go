package onnxruntime

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api"
)

// TensorData is a type constraint for supported tensor data types.
// It includes all numeric types, bool, and complex types that are supported by ONNX Runtime.
type TensorData interface {
	~float32 | ~float64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~bool
}

// Value represents an ONNX Runtime value, typically a tensor.
// Values are used as inputs and outputs for model inference.
type Value struct {
	ptr     api.OrtValue
	infoPtr api.OrtTensorTypeAndShapeInfo
	runtime *Runtime
	cleanup     runtime.Cleanup // handle to cancel GC cleanup of OrtValue
	infoCleanup runtime.Cleanup // handle to cancel GC cleanup of TensorTypeAndShapeInfo
}

func (r *Runtime) newValueFromPtr(ptr api.OrtValue) *Value {
	v := &Value{
		ptr:     ptr,
		runtime: r,
	}

	// Clean up the native OrtValue when *Value is no longer reachable.
	// Must not close over v (would prevent GC, panics on Go 1.24+).
	v.cleanup = runtime.AddCleanup(v, r.releaseValuePtr, ptr)
	return v
}

func (v *Value) initTensorTypeAndShapeInfo() error {
	if v.infoPtr != 0 {
		// already initialized
		return nil
	}

	var infoPtr api.OrtTensorTypeAndShapeInfo
	status := v.runtime.apiFuncs.GetTensorTypeAndShape(v.ptr, &infoPtr)
	if err := v.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to get tensor type and shape: %w", err)
	}
	v.infoPtr = infoPtr
	v.infoCleanup = runtime.AddCleanup(v, v.runtime.releaseInfoPtr, infoPtr)
	return nil
}

// getTensorMutableData returns a pointer to the tensor's underlying data buffer.
func (v *Value) getTensorMutableData() (unsafe.Pointer, error) {
	var dataPtr unsafe.Pointer
	status := v.runtime.apiFuncs.GetTensorMutableData(v.ptr, &dataPtr)
	if err := v.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get tensor data: %w", err)
	}

	return dataPtr, nil
}

// GetValueType returns the type of the value (tensor, sequence, map, etc.).
func (v *Value) GetValueType() (ONNXType, error) {
	var valueType ONNXType
	status := v.runtime.apiFuncs.GetValueType(v.ptr, &valueType)
	if err := v.runtime.statusError(status); err != nil {
		return ONNXTypeUnknown, fmt.Errorf("failed to get value type: %w", err)
	}

	return valueType, nil
}

// GetTensorShape returns the shape (dimensions) of the tensor as a slice of int64 values.
// For example, a 2x3 matrix returns [2, 3].
func (v *Value) GetTensorShape() ([]int64, error) {
	if err := v.initTensorTypeAndShapeInfo(); err != nil {
		return nil, err
	}

	// Get dimension count
	var dimCount uintptr
	status := v.runtime.apiFuncs.GetDimensionsCount(v.infoPtr, &dimCount)
	if err := v.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	// Get dimensions
	dims := make([]int64, dimCount)
	if dimCount > 0 {
		status = v.runtime.apiFuncs.GetDimensions(v.infoPtr, &dims[0], dimCount)
		if err := v.runtime.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to get dimensions: %w", err)
		}
	}

	return dims, nil
}

// GetTensorElementType returns the data type of the tensor's elements.
func (v *Value) GetTensorElementType() (ONNXTensorElementDataType, error) {
	if err := v.initTensorTypeAndShapeInfo(); err != nil {
		return ONNXTensorElementDataTypeUndefined, err
	}

	var elemType ONNXTensorElementDataType
	status := v.runtime.apiFuncs.GetTensorElementType(v.infoPtr, &elemType)
	if err := v.runtime.statusError(status); err != nil {
		return ONNXTensorElementDataTypeUndefined, fmt.Errorf("failed to get element type: %w", err)
	}

	return elemType, nil
}

// GetElementCount returns the total number of elements in the tensor.
// For example, a 2x3 matrix has 6 elements.
func (v *Value) GetTensorElementCount() (int, error) {
	if err := v.initTensorTypeAndShapeInfo(); err != nil {
		return 0, err
	}

	var count uintptr
	status := v.runtime.apiFuncs.GetTensorShapeElementCount(v.infoPtr, &count)
	if err := v.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get element count: %w", err)
	}

	return int(count), nil
}

// Close releases the value and associated resources.
// It is safe to call Close multiple times.
//
// While a finalizer is set as a safety net to automatically release resources
// when the Value is garbage collected, explicitly calling Close is strongly
// recommended to ensure timely release of native memory, especially when
// dealing with large tensors or high-frequency inference operations.
func (v *Value) Close() {
	v.cleanup.Stop()
	v.infoCleanup.Stop()
	v.releaseValue()
	v.releaseInfo()
}

func (r *Runtime) releaseValuePtr(p api.OrtValue) {
	if p != 0 && r.apiFuncs != nil {
		r.apiFuncs.ReleaseValue(p)
	}
}

func (r *Runtime) releaseInfoPtr(p api.OrtTensorTypeAndShapeInfo) {
	if p != 0 && r.apiFuncs != nil {
		r.apiFuncs.ReleaseTensorTypeAndShapeInfo(p)
	}
}

func (v *Value) releaseValue() {
	if v.ptr != 0 && v.runtime != nil {
		v.runtime.releaseValuePtr(v.ptr)
		v.ptr = 0
	}
}

func (v *Value) releaseInfo() {
	if v.infoPtr != 0 && v.runtime != nil {
		v.runtime.releaseInfoPtr(v.infoPtr)
		v.infoPtr = 0
	}
}

// NewTensorValue creates a new tensor value from a slice of data using type inference.
// This is a generic function that supports all numeric types and bool via the TensorData constraint.
// The data slice must not be empty, and the shape defines the tensor dimensions.
func NewTensorValue[T TensorData](r *Runtime, data []T, shape []int64) (*Value, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("data cannot be empty")
	}

	var dataType ONNXTensorElementDataType
	var elementSize uintptr

	// Determine data type and element size based on the type parameter
	var zero T
	switch any(zero).(type) {
	case float32:
		dataType = ONNXTensorElementDataTypeFloat
		elementSize = 4
	case float64:
		dataType = ONNXTensorElementDataTypeDouble
		elementSize = 8
	case int8:
		dataType = ONNXTensorElementDataTypeInt8
		elementSize = 1
	case int16:
		dataType = ONNXTensorElementDataTypeInt16
		elementSize = 2
	case int32:
		dataType = ONNXTensorElementDataTypeInt32
		elementSize = 4
	case int64:
		dataType = ONNXTensorElementDataTypeInt64
		elementSize = 8
	case uint8:
		dataType = ONNXTensorElementDataTypeUint8
		elementSize = 1
	case uint16:
		dataType = ONNXTensorElementDataTypeUint16
		elementSize = 2
	case uint32:
		dataType = ONNXTensorElementDataTypeUint32
		elementSize = 4
	case uint64:
		dataType = ONNXTensorElementDataTypeUint64
		elementSize = 8
	case bool:
		dataType = ONNXTensorElementDataTypeBool
		elementSize = 1
	default:
		return nil, fmt.Errorf("unsupported data type")
	}

	dataPtr := unsafe.Pointer(&data[0])
	dataLen := uintptr(len(data)) * elementSize

	return r.newTensorValue(dataPtr, dataLen, shape, dataType)
}

// newTensorValue creates a new tensor value from raw data using default CPU memory.
// The data pointer must point to contiguous memory of size dataLen bytes.
// The shape defines the tensor dimensions, and dataType specifies the element type.
func (r *Runtime) newTensorValue(data unsafe.Pointer, dataLen uintptr, shape []int64, dataType ONNXTensorElementDataType) (*Value, error) {
	if r.cpuMemoryInfo == nil {
		return nil, fmt.Errorf("default memory info not initialized")
	}

	var valuePtr api.OrtValue
	var shapePtr *int64
	if len(shape) > 0 {
		shapePtr = &shape[0]
	}

	status := r.apiFuncs.CreateTensorWithDataAsOrtValue(r.cpuMemoryInfo.ptr, data, dataLen, shapePtr, uintptr(len(shape)), dataType, &valuePtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create tensor: %w", err)
	}
	return r.newValueFromPtr(valuePtr), nil
}

// GetTensorData extracts tensor data and shape from a Value.
// This is a generic function that supports all numeric types and bool via the TensorData constraint.
// It returns both the data as a slice and the shape of the tensor.
// The returned data slice is a copy of the tensor data.
func GetTensorData[T TensorData](v *Value) ([]T, []int64, error) {
	// Get shape first
	shape, err := v.GetTensorShape()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get shape: %w", err)
	}

	// Verify element type matches expected type T
	elemType, err := v.GetTensorElementType()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get element type: %w", err)
	}

	var zero T
	var expectedType ONNXTensorElementDataType
	switch any(zero).(type) {
	case float32:
		expectedType = ONNXTensorElementDataTypeFloat
	case float64:
		expectedType = ONNXTensorElementDataTypeDouble
	case int8:
		expectedType = ONNXTensorElementDataTypeInt8
	case int16:
		expectedType = ONNXTensorElementDataTypeInt16
	case int32:
		expectedType = ONNXTensorElementDataTypeInt32
	case int64:
		expectedType = ONNXTensorElementDataTypeInt64
	case uint8:
		expectedType = ONNXTensorElementDataTypeUint8
	case uint16:
		expectedType = ONNXTensorElementDataTypeUint16
	case uint32:
		expectedType = ONNXTensorElementDataTypeUint32
	case uint64:
		expectedType = ONNXTensorElementDataTypeUint64
	case bool:
		expectedType = ONNXTensorElementDataTypeBool
	}

	if elemType != expectedType {
		return nil, nil, fmt.Errorf("element type mismatch: expected %d, got %d", expectedType, elemType)
	}

	dataPtr, err := v.getTensorMutableData()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get tensor data: %w", err)
	}

	// Get element count
	count, err := v.GetTensorElementCount()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get element count: %w", err)
	}

	// Create a slice backed by the tensor's data
	data := unsafe.Slice((*T)(dataPtr), count)

	// Make a copy to avoid issues with the underlying memory
	result := make([]T, count)
	copy(result, data)

	return result, shape, nil
}
