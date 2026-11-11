package onnxruntime

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api"
)

// Env represents an ONNX Runtime environment that manages global state
// and configuration for all inference sessions.
type Env struct {
	ptr     api.OrtEnv
	runtime *Runtime
	// logger is the registry handle of the LogFunc this env was created
	// with by NewEnvWithLogger; 0 for an env created by NewEnv.
	logger uintptr
}

// LogFunc receives one line ONNX Runtime would otherwise print: its
// severity, category, the env's log identifier, the C++ code location and
// the message. It is called on ONNX Runtime's own threads.
type LogFunc func(severity LoggingLevel, category, logID, codeLocation, message string)

// loggers holds every live LogFunc by the opaque handle ONNX Runtime calls
// the one C callback back with, so a Go function pointer never crosses the
// C boundary and a closed env's function is forgotten.
var (
	loggers      sync.Map // uintptr -> LogFunc
	nextLogger   atomic.Uintptr
	callbackOnce sync.Once
	callbackPtr  uintptr
)

// loggingCallback is the one OrtLoggingFunction this process registers with
// ONNX Runtime: purego allows a bounded number of callbacks per process, so
// every env shares it and is told apart by its param.
func loggingCallback() uintptr {
	callbackOnce.Do(func() {
		callbackPtr = purego.NewCallback(func(param uintptr, severity int32, category, logID, codeLocation, message *byte) uintptr {
			if fn, ok := loggers.Load(param); ok {
				fn.(LogFunc)(LoggingLevel(severity), cString(category), cString(logID), cString(codeLocation), cString(message))
			}
			return 0
		})
	})
	return callbackPtr
}

// cString reads a NUL-terminated C string.
func cString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

// NewEnv creates a new ONNX Runtime environment with the specified logging level and identifier.
// The logLevel parameter controls logging verbosity, and logID is used to tag log messages.
func (r *Runtime) NewEnv(logID string, logLevel LoggingLevel) (*Env, error) {
	logIDBytes := append([]byte(logID), 0)
	var envPtr api.OrtEnv

	status := r.apiFuncs.CreateEnv(logLevel, &logIDBytes[0], &envPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	return &Env{
		ptr:     envPtr,
		runtime: r,
	}, nil
}

// NewEnvWithLogger creates an environment whose log lines go to fn instead
// of ONNX Runtime's default stderr logger (OrtApi::CreateEnvWithCustomLogger).
// logLevel is the least severity fn is called for; SetLogLevel changes it.
func (r *Runtime) NewEnvWithLogger(logID string, logLevel LoggingLevel, fn LogFunc) (*Env, error) {
	if fn == nil {
		return nil, fmt.Errorf("failed to create environment: nil LogFunc")
	}
	logIDBytes := append([]byte(logID), 0)
	var envPtr api.OrtEnv
	handle := nextLogger.Add(1)
	loggers.Store(handle, fn)

	status := r.apiFuncs.CreateEnvWithCustomLogger(loggingCallback(), handle, logLevel, &logIDBytes[0], &envPtr)
	if err := r.statusError(status); err != nil {
		loggers.Delete(handle)
		return nil, fmt.Errorf("failed to create environment: %w", err)
	}

	return &Env{
		ptr:     envPtr,
		runtime: r,
		logger:  handle,
	}, nil
}

// SetLogLevel changes the least severity the environment logs
// (OrtApi::UpdateEnvWithCustomLogLevel), for its own logger and a custom one
// alike.
func (e *Env) SetLogLevel(logLevel LoggingLevel) error {
	if e.ptr == 0 || e.runtime == nil || e.runtime.apiFuncs == nil {
		return fmt.Errorf("failed to set log level: environment is closed")
	}
	status := e.runtime.apiFuncs.UpdateEnvWithCustomLogLevel(e.ptr, logLevel)
	if err := e.runtime.statusError(status); err != nil {
		return fmt.Errorf("failed to set log level: %w", err)
	}
	return nil
}

// Close releases the environment and frees associated resources.
func (e *Env) Close() {
	if e.ptr != 0 && e.runtime != nil && e.runtime.apiFuncs != nil {
		e.runtime.apiFuncs.ReleaseEnv(e.ptr)
		e.ptr = 0
	}
	if e.logger != 0 {
		loggers.Delete(e.logger)
		e.logger = 0
	}
}
