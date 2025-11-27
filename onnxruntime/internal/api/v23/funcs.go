package v23

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api"
)

// Funcs contains cached function pointers to ONNX Runtime C API functions.
type Funcs struct {
	// Status and error handling
	createStatus    func(api.OrtErrorCode, *byte) api.OrtStatus
	getErrorCode    func(api.OrtStatus) api.OrtErrorCode
	getErrorMessage func(api.OrtStatus) unsafe.Pointer
	releaseStatus   func(api.OrtStatus)

	// Environment
	createEnv  func(api.OrtLoggingLevel, *byte, *api.OrtEnv) api.OrtStatus
	releaseEnv func(api.OrtEnv)

	// Allocator
	getAllocatorWithDefaultOptions func(*api.OrtAllocator) api.OrtStatus
	allocatorFree                  func(api.OrtAllocator, unsafe.Pointer)

	// Memory info
	createCpuMemoryInfo func(api.OrtAllocatorType, api.OrtMemType, *api.OrtMemoryInfo) api.OrtStatus
	releaseMemoryInfo   func(api.OrtMemoryInfo)

	// Session options
	createSessionOptions                  func(*api.OrtSessionOptions) api.OrtStatus
	setIntraOpNumThreads                  func(api.OrtSessionOptions, int32) api.OrtStatus
	sessionOptionsAppendExecutionProvider func(api.OrtSessionOptions, *byte, **byte, **byte, uintptr) api.OrtStatus
	releaseSessionOptions                 func(api.OrtSessionOptions)

	// Session
	createSession          func(api.OrtEnv, *byte, api.OrtSessionOptions, *api.OrtSession) api.OrtStatus
	createSessionFromArray func(api.OrtEnv, unsafe.Pointer, uintptr, api.OrtSessionOptions, *api.OrtSession) api.OrtStatus
	sessionGetInputCount   func(api.OrtSession, *uintptr) api.OrtStatus
	sessionGetOutputCount  func(api.OrtSession, *uintptr) api.OrtStatus
	sessionGetInputName    func(api.OrtSession, uintptr, api.OrtAllocator, **byte) api.OrtStatus
	sessionGetOutputName   func(api.OrtSession, uintptr, api.OrtAllocator, **byte) api.OrtStatus
	run                    func(api.OrtSession, uintptr, **byte, *api.OrtValue, uintptr, **byte, uintptr, *api.OrtValue) api.OrtStatus
	releaseSession         func(api.OrtSession)

	// Tensor/Value operations
	createTensorWithDataAsOrtValue func(api.OrtMemoryInfo, unsafe.Pointer, uintptr, *int64, uintptr, api.ONNXTensorElementDataType, *api.OrtValue) api.OrtStatus
	getValueType                   func(api.OrtValue, *api.ONNXType) api.OrtStatus
	getTensorMutableData           func(api.OrtValue, *unsafe.Pointer) api.OrtStatus
	getTensorTypeAndShape          func(api.OrtValue, *api.OrtTensorTypeAndShapeInfo) api.OrtStatus
	getTensorElementType           func(api.OrtTensorTypeAndShapeInfo, *api.ONNXTensorElementDataType) api.OrtStatus
	getDimensionsCount             func(api.OrtTensorTypeAndShapeInfo, *uintptr) api.OrtStatus
	getDimensions                  func(api.OrtTensorTypeAndShapeInfo, *int64, uintptr) api.OrtStatus
	getTensorShapeElementCount     func(api.OrtTensorTypeAndShapeInfo, *uintptr) api.OrtStatus
	releaseValue                   func(api.OrtValue)
	releaseTensorTypeAndShapeInfo  func(api.OrtTensorTypeAndShapeInfo)

	// Execution provider information
	getAvailableProviders     func(***byte, *int32) api.OrtStatus
	releaseAvailableProviders func(**byte, int32) api.OrtStatus
}

// InitializeFuncs initializes the v23 API function pointers from the library handle.
// This is called once during initialization to avoid repeated RegisterFunc calls.
func InitializeFuncs(libraryHandle uintptr) (*Funcs, error) {
	// Get the OrtApiBase from the library
	var ortGetAPIBase func() *APIBase
	purego.RegisterLibFunc(&ortGetAPIBase, libraryHandle, "OrtGetApiBase")

	apiBase := ortGetAPIBase()
	if apiBase == nil {
		return nil, fmt.Errorf("OrtGetApiBase returned nil")
	}

	// Get the versioned API
	var getAPIFunc func(uint32) unsafe.Pointer
	purego.RegisterFunc(&getAPIFunc, apiBase.GetAPI)

	apiPtr := getAPIFunc(APIVersion)
	if apiPtr == nil {
		return nil, fmt.Errorf("failed to get OrtAPI for version %d", APIVersion)
	}

	api := (*API)(apiPtr)

	funcs := &Funcs{}

	// Register all function pointers
	purego.RegisterFunc(&funcs.createStatus, api.CreateStatus)
	purego.RegisterFunc(&funcs.getErrorCode, api.GetErrorCode)
	purego.RegisterFunc(&funcs.getErrorMessage, api.GetErrorMessage)
	purego.RegisterFunc(&funcs.releaseStatus, api.ReleaseStatus)

	purego.RegisterFunc(&funcs.createEnv, api.CreateEnv)
	purego.RegisterFunc(&funcs.releaseEnv, api.ReleaseEnv)

	purego.RegisterFunc(&funcs.getAllocatorWithDefaultOptions, api.GetAllocatorWithDefaultOptions)
	purego.RegisterFunc(&funcs.allocatorFree, api.AllocatorFree)

	purego.RegisterFunc(&funcs.createCpuMemoryInfo, api.CreateCpuMemoryInfo)
	purego.RegisterFunc(&funcs.releaseMemoryInfo, api.ReleaseMemoryInfo)

	purego.RegisterFunc(&funcs.createSessionOptions, api.CreateSessionOptions)
	purego.RegisterFunc(&funcs.setIntraOpNumThreads, api.SetIntraOpNumThreads)
	purego.RegisterFunc(&funcs.sessionOptionsAppendExecutionProvider, api.SessionOptionsAppendExecutionProvider)
	purego.RegisterFunc(&funcs.releaseSessionOptions, api.ReleaseSessionOptions)

	purego.RegisterFunc(&funcs.createSession, api.CreateSession)
	purego.RegisterFunc(&funcs.createSessionFromArray, api.CreateSessionFromArray)
	purego.RegisterFunc(&funcs.sessionGetInputCount, api.SessionGetInputCount)
	purego.RegisterFunc(&funcs.sessionGetOutputCount, api.SessionGetOutputCount)
	purego.RegisterFunc(&funcs.sessionGetInputName, api.SessionGetInputName)
	purego.RegisterFunc(&funcs.sessionGetOutputName, api.SessionGetOutputName)
	purego.RegisterFunc(&funcs.run, api.Run)
	purego.RegisterFunc(&funcs.releaseSession, api.ReleaseSession)

	purego.RegisterFunc(&funcs.createTensorWithDataAsOrtValue, api.CreateTensorWithDataAsOrtValue)
	purego.RegisterFunc(&funcs.getValueType, api.GetValueType)
	purego.RegisterFunc(&funcs.getTensorMutableData, api.GetTensorMutableData)
	purego.RegisterFunc(&funcs.getTensorTypeAndShape, api.GetTensorTypeAndShape)
	purego.RegisterFunc(&funcs.getTensorElementType, api.GetTensorElementType)
	purego.RegisterFunc(&funcs.getDimensionsCount, api.GetDimensionsCount)
	purego.RegisterFunc(&funcs.getDimensions, api.GetDimensions)
	purego.RegisterFunc(&funcs.getTensorShapeElementCount, api.GetTensorShapeElementCount)
	purego.RegisterFunc(&funcs.releaseValue, api.ReleaseValue)
	purego.RegisterFunc(&funcs.releaseTensorTypeAndShapeInfo, api.ReleaseTensorTypeAndShapeInfo)

	purego.RegisterFunc(&funcs.getAvailableProviders, api.GetAvailableProviders)
	purego.RegisterFunc(&funcs.releaseAvailableProviders, api.ReleaseAvailableProviders)

	return funcs, nil
}

// Status and error handling methods

func (f *Funcs) CreateStatus(code api.OrtErrorCode, msg *byte) api.OrtStatus {
	return f.createStatus(code, msg)
}

func (f *Funcs) GetErrorCode(status api.OrtStatus) api.OrtErrorCode {
	return f.getErrorCode(status)
}

func (f *Funcs) GetErrorMessage(status api.OrtStatus) unsafe.Pointer {
	return f.getErrorMessage(status)
}

func (f *Funcs) ReleaseStatus(status api.OrtStatus) {
	f.releaseStatus(status)
}

// Environment methods

func (f *Funcs) CreateEnv(logLevel api.OrtLoggingLevel, logID *byte, env *api.OrtEnv) api.OrtStatus {
	return f.createEnv(logLevel, logID, env)
}

func (f *Funcs) ReleaseEnv(env api.OrtEnv) {
	f.releaseEnv(env)
}

// Allocator methods

func (f *Funcs) GetAllocatorWithDefaultOptions(allocator *api.OrtAllocator) api.OrtStatus {
	return f.getAllocatorWithDefaultOptions(allocator)
}

func (f *Funcs) AllocatorFree(allocator api.OrtAllocator, ptr unsafe.Pointer) {
	f.allocatorFree(allocator, ptr)
}

// Memory info methods

func (f *Funcs) CreateCpuMemoryInfo(allocType api.OrtAllocatorType, memType api.OrtMemType, memInfo *api.OrtMemoryInfo) api.OrtStatus {
	return f.createCpuMemoryInfo(allocType, memType, memInfo)
}

func (f *Funcs) ReleaseMemoryInfo(memInfo api.OrtMemoryInfo) {
	f.releaseMemoryInfo(memInfo)
}

// Session options methods

func (f *Funcs) CreateSessionOptions(options *api.OrtSessionOptions) api.OrtStatus {
	return f.createSessionOptions(options)
}

func (f *Funcs) SetIntraOpNumThreads(options api.OrtSessionOptions, numThreads int32) api.OrtStatus {
	return f.setIntraOpNumThreads(options, numThreads)
}

func (f *Funcs) SessionOptionsAppendExecutionProvider(options api.OrtSessionOptions, providerName *byte, keys **byte, values **byte, numKeys uintptr) api.OrtStatus {
	return f.sessionOptionsAppendExecutionProvider(options, providerName, keys, values, numKeys)
}

func (f *Funcs) ReleaseSessionOptions(options api.OrtSessionOptions) {
	f.releaseSessionOptions(options)
}

// Session methods

func (f *Funcs) CreateSession(env api.OrtEnv, modelPath *byte, options api.OrtSessionOptions, session *api.OrtSession) api.OrtStatus {
	return f.createSession(env, modelPath, options, session)
}

func (f *Funcs) CreateSessionFromArray(env api.OrtEnv, modelData unsafe.Pointer, modelDataLength uintptr, options api.OrtSessionOptions, session *api.OrtSession) api.OrtStatus {
	return f.createSessionFromArray(env, modelData, modelDataLength, options, session)
}

func (f *Funcs) SessionGetInputCount(session api.OrtSession, count *uintptr) api.OrtStatus {
	return f.sessionGetInputCount(session, count)
}

func (f *Funcs) SessionGetOutputCount(session api.OrtSession, count *uintptr) api.OrtStatus {
	return f.sessionGetOutputCount(session, count)
}

func (f *Funcs) SessionGetInputName(session api.OrtSession, index uintptr, allocator api.OrtAllocator, name **byte) api.OrtStatus {
	return f.sessionGetInputName(session, index, allocator, name)
}

func (f *Funcs) SessionGetOutputName(session api.OrtSession, index uintptr, allocator api.OrtAllocator, name **byte) api.OrtStatus {
	return f.sessionGetOutputName(session, index, allocator, name)
}

func (f *Funcs) Run(session api.OrtSession, runOptions uintptr, inputNames **byte, inputs *api.OrtValue, inputCount uintptr, outputNames **byte, outputCount uintptr, outputs *api.OrtValue) api.OrtStatus {
	return f.run(session, runOptions, inputNames, inputs, inputCount, outputNames, outputCount, outputs)
}

func (f *Funcs) ReleaseSession(session api.OrtSession) {
	f.releaseSession(session)
}

// Tensor/Value operations methods

func (f *Funcs) CreateTensorWithDataAsOrtValue(memInfo api.OrtMemoryInfo, data unsafe.Pointer, dataSize uintptr, shape *int64, shapeLen uintptr, dataType api.ONNXTensorElementDataType, value *api.OrtValue) api.OrtStatus {
	return f.createTensorWithDataAsOrtValue(memInfo, data, dataSize, shape, shapeLen, dataType, value)
}

func (f *Funcs) GetValueType(value api.OrtValue, valueType *api.ONNXType) api.OrtStatus {
	return f.getValueType(value, valueType)
}

func (f *Funcs) GetTensorMutableData(value api.OrtValue, data *unsafe.Pointer) api.OrtStatus {
	return f.getTensorMutableData(value, data)
}

func (f *Funcs) GetTensorTypeAndShape(value api.OrtValue, typeAndShape *api.OrtTensorTypeAndShapeInfo) api.OrtStatus {
	return f.getTensorTypeAndShape(value, typeAndShape)
}

func (f *Funcs) GetTensorElementType(typeAndShape api.OrtTensorTypeAndShapeInfo, dataType *api.ONNXTensorElementDataType) api.OrtStatus {
	return f.getTensorElementType(typeAndShape, dataType)
}

func (f *Funcs) GetDimensionsCount(typeAndShape api.OrtTensorTypeAndShapeInfo, count *uintptr) api.OrtStatus {
	return f.getDimensionsCount(typeAndShape, count)
}

func (f *Funcs) GetDimensions(typeAndShape api.OrtTensorTypeAndShapeInfo, dims *int64, dimsLen uintptr) api.OrtStatus {
	return f.getDimensions(typeAndShape, dims, dimsLen)
}

func (f *Funcs) GetTensorShapeElementCount(typeAndShape api.OrtTensorTypeAndShapeInfo, count *uintptr) api.OrtStatus {
	return f.getTensorShapeElementCount(typeAndShape, count)
}

func (f *Funcs) ReleaseValue(value api.OrtValue) {
	f.releaseValue(value)
}

func (f *Funcs) ReleaseTensorTypeAndShapeInfo(typeAndShape api.OrtTensorTypeAndShapeInfo) {
	f.releaseTensorTypeAndShapeInfo(typeAndShape)
}

// Execution provider information methods

func (f *Funcs) GetAvailableProviders(providers ***byte, length *int32) api.OrtStatus {
	return f.getAvailableProviders(providers, length)
}

func (f *Funcs) ReleaseAvailableProviders(providers **byte, length int32) api.OrtStatus {
	return f.releaseAvailableProviders(providers, length)
}
