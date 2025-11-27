package api

import "unsafe"

// OrtStatus is an opaque pointer to an ONNX Runtime status object.
type OrtStatus uintptr

// OrtEnv is an opaque pointer to an ONNX Runtime environment.
type OrtEnv uintptr

// OrtSession is an opaque pointer to an ONNX Runtime inference session.
type OrtSession uintptr

// OrtSessionOptions is an opaque pointer to ONNX Runtime session options.
type OrtSessionOptions uintptr

// OrtValue is an opaque pointer to an ONNX Runtime value (typically a tensor).
type OrtValue uintptr

// OrtAllocator is an opaque pointer to an ONNX Runtime memory allocator.
type OrtAllocator uintptr

// OrtMemoryInfo is an opaque pointer to ONNX Runtime memory information.
type OrtMemoryInfo uintptr

// OrtTensorTypeAndShapeInfo is an opaque pointer to ONNX Runtime tensor type and shape information.
type OrtTensorTypeAndShapeInfo uintptr

// OrtErrorCode represents error codes returned by the ONNX Runtime C API.
type OrtErrorCode int32

// OrtLoggingLevel represents logging verbosity levels for ONNX Runtime.
type OrtLoggingLevel int32

// ONNXType represents the type of an ONNX value.
type ONNXType int32

// ONNXTensorElementDataType represents the data type of tensor elements.
type ONNXTensorElementDataType int32

// OrtAllocatorType represents memory allocator types.
type OrtAllocatorType int32

// OrtMemType represents memory types for allocations.
type OrtMemType int32

// APIFuncs is an interface for ONNX Runtime C API functions.
type APIFuncs interface {
	// Status and error handling
	CreateStatus(OrtErrorCode, *byte) OrtStatus
	GetErrorCode(OrtStatus) OrtErrorCode
	GetErrorMessage(OrtStatus) unsafe.Pointer
	ReleaseStatus(OrtStatus)

	// Environment
	CreateEnv(OrtLoggingLevel, *byte, *OrtEnv) OrtStatus
	ReleaseEnv(OrtEnv)

	// Allocator
	GetAllocatorWithDefaultOptions(*OrtAllocator) OrtStatus
	AllocatorFree(OrtAllocator, unsafe.Pointer)

	// Memory info
	CreateCpuMemoryInfo(OrtAllocatorType, OrtMemType, *OrtMemoryInfo) OrtStatus
	ReleaseMemoryInfo(OrtMemoryInfo)

	// Session options
	CreateSessionOptions(*OrtSessionOptions) OrtStatus
	SetIntraOpNumThreads(OrtSessionOptions, int32) OrtStatus
	SessionOptionsAppendExecutionProvider(OrtSessionOptions, *byte, **byte, **byte, uintptr) OrtStatus
	ReleaseSessionOptions(OrtSessionOptions)

	// Session
	CreateSession(OrtEnv, *byte, OrtSessionOptions, *OrtSession) OrtStatus
	CreateSessionFromArray(OrtEnv, unsafe.Pointer, uintptr, OrtSessionOptions, *OrtSession) OrtStatus
	SessionGetInputCount(OrtSession, *uintptr) OrtStatus
	SessionGetOutputCount(OrtSession, *uintptr) OrtStatus
	SessionGetInputName(OrtSession, uintptr, OrtAllocator, **byte) OrtStatus
	SessionGetOutputName(OrtSession, uintptr, OrtAllocator, **byte) OrtStatus
	Run(OrtSession, uintptr, **byte, *OrtValue, uintptr, **byte, uintptr, *OrtValue) OrtStatus
	ReleaseSession(OrtSession)

	// Tensor/Value operations
	CreateTensorWithDataAsOrtValue(OrtMemoryInfo, unsafe.Pointer, uintptr, *int64, uintptr, ONNXTensorElementDataType, *OrtValue) OrtStatus
	GetValueType(OrtValue, *ONNXType) OrtStatus
	GetTensorMutableData(OrtValue, *unsafe.Pointer) OrtStatus
	GetTensorTypeAndShape(OrtValue, *OrtTensorTypeAndShapeInfo) OrtStatus
	GetTensorElementType(OrtTensorTypeAndShapeInfo, *ONNXTensorElementDataType) OrtStatus
	GetDimensionsCount(OrtTensorTypeAndShapeInfo, *uintptr) OrtStatus
	GetDimensions(OrtTensorTypeAndShapeInfo, *int64, uintptr) OrtStatus
	GetTensorShapeElementCount(OrtTensorTypeAndShapeInfo, *uintptr) OrtStatus
	ReleaseValue(OrtValue)
	ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo)

	// Execution provider information
	GetAvailableProviders(***byte, *int32) OrtStatus
	ReleaseAvailableProviders(**byte, int32) OrtStatus
}
