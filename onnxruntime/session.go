package onnxruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"unsafe"

	"github.com/shota3506/onnxruntime-purego/internal/cstrings"
	"github.com/shota3506/onnxruntime-purego/onnxruntime/internal/api"
)

// SessionOptions configures options for creating an inference session.
type SessionOptions struct {
	// IntraOpNumThreads sets the number of threads used for parallelizing
	// execution within nodes. A value of 0 uses the default number of threads.
	IntraOpNumThreads int

	// ExecutionProviders specifies the execution providers to use, in order of preference.
	// Common values include "CPUExecutionProvider", "CUDAExecutionProvider", etc.
	// If empty, the default provider(s) will be used.
	ExecutionProviders []string
}

// Session represents an ONNX Runtime inference session that can execute
// a loaded ONNX model.
type Session struct {
	ptr     api.OrtSession
	runtime *Runtime

	// metadata
	inputNames  []string
	outputNames []string
}

// NewSession creates a new inference session from a model file.
func (r *Runtime) NewSession(env *Env, modelPath string, options *SessionOptions) (*Session, error) {
	var optsPtr api.OrtSessionOptions
	if options != nil {
		status := r.apiFuncs.CreateSessionOptions(&optsPtr)
		if err := r.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to create session options: %w", err)
		}
		defer func() {
			if optsPtr != 0 {
				r.apiFuncs.ReleaseSessionOptions(optsPtr)
			}
		}()

		if err := r.configureIntraOpNumThreads(optsPtr, options); err != nil {
			return nil, fmt.Errorf("failed to configure intra-op num threads: %w", err)
		}
		if err := r.configureExecutionProviders(optsPtr, options); err != nil {
			return nil, fmt.Errorf("failed to configure execution providers: %w", err)
		}
	}

	modelPathBytes := append([]byte(modelPath), 0)
	var sessionPtr api.OrtSession

	status := r.apiFuncs.CreateSession(env.ptr, &modelPathBytes[0], api.OrtSessionOptions(optsPtr), &sessionPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	session := &Session{
		ptr:     sessionPtr,
		runtime: r,
	}

	// Initialize metadata cache
	if err := session.initializeMetadata(); err != nil {
		session.Close()
		return nil, fmt.Errorf("failed to initialize session metadata: %w", err)
	}

	return session, nil
}

// NewSessionFromReader creates a new inference session from a model loaded from modelReader.
// The modelReader contains the ONNX model data, and options configures session-specific settings (may be nil for defaults).
func (r *Runtime) NewSessionFromReader(env *Env, modelReader io.Reader, options *SessionOptions) (*Session, error) {
	// Read all data from the reader
	modelData, err := io.ReadAll(modelReader)
	if err != nil {
		return nil, fmt.Errorf("failed to read model data: %w", err)
	}

	if len(modelData) == 0 {
		return nil, fmt.Errorf("model data cannot be empty")
	}

	var optsPtr api.OrtSessionOptions
	if options != nil {
		status := r.apiFuncs.CreateSessionOptions(&optsPtr)
		if err := r.statusError(status); err != nil {
			return nil, fmt.Errorf("failed to create session options: %w", err)
		}
		defer func() {
			if optsPtr != 0 {
				r.apiFuncs.ReleaseSessionOptions(optsPtr)
			}
		}()

		if err := r.configureIntraOpNumThreads(optsPtr, options); err != nil {
			return nil, fmt.Errorf("failed to configure intra-op num threads: %w", err)
		}
		if err := r.configureExecutionProviders(optsPtr, options); err != nil {
			return nil, fmt.Errorf("failed to configure execution providers: %w", err)
		}
	}

	var sessionPtr api.OrtSession

	status := r.apiFuncs.CreateSessionFromArray(env.ptr, unsafe.Pointer(&modelData[0]), uintptr(len(modelData)), api.OrtSessionOptions(optsPtr), &sessionPtr)
	if err := r.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	session := &Session{
		ptr:     sessionPtr,
		runtime: r,
	}

	// Initialize metadata cache
	if err := session.initializeMetadata(); err != nil {
		session.Close()
		return nil, fmt.Errorf("failed to initialize session metadata: %w", err)
	}

	return session, nil
}

// initializeMetadata caches input and output names during session creation
func (s *Session) initializeMetadata() error {
	// Get input count and names
	inputCount, err := s.getInputCount()
	if err != nil {
		return fmt.Errorf("failed to get input count: %w", err)
	}

	s.inputNames = make([]string, inputCount)
	for i := range inputCount {
		name, err := s.getInputName(i)
		if err != nil {
			return fmt.Errorf("failed to get input name at index %d: %w", i, err)
		}
		s.inputNames[i] = name
	}

	// Get output count and names
	outputCount, err := s.getOutputCount()
	if err != nil {
		return fmt.Errorf("failed to get output count: %w", err)
	}

	s.outputNames = make([]string, outputCount)
	for i := range outputCount {
		name, err := s.getOutputName(i)
		if err != nil {
			return fmt.Errorf("failed to get output name at index %d: %w", i, err)
		}
		s.outputNames[i] = name
	}

	return nil
}

// InputNames returns all input names for the model.
func (s *Session) InputNames() []string {
	return s.inputNames
}

// OutputNames returns all output names for the model.
func (s *Session) OutputNames() []string {
	return s.outputNames
}

// getInputCount retrieves the input count from ONNX Runtime (internal use)
func (s *Session) getInputCount() (int, error) {
	if s.ptr == 0 {
		return 0, ErrSessionClosed
	}

	var count uintptr
	status := s.runtime.apiFuncs.SessionGetInputCount(s.ptr, &count)
	if err := s.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get input count: %w", err)
	}

	return int(count), nil
}

// getOutputCount retrieves the output count from ONNX Runtime (internal use)
func (s *Session) getOutputCount() (int, error) {
	if s.ptr == 0 {
		return 0, ErrSessionClosed
	}

	var count uintptr
	status := s.runtime.apiFuncs.SessionGetOutputCount(s.ptr, &count)
	if err := s.runtime.statusError(status); err != nil {
		return 0, fmt.Errorf("failed to get output count: %w", err)
	}

	return int(count), nil
}

// getInputName retrieves the input name from ONNX Runtime (internal use)
func (s *Session) getInputName(index int) (string, error) {
	if s.ptr == 0 {
		return "", ErrSessionClosed
	}

	if s.runtime.allocator == nil {
		return "", errors.New("allocator not initialized")
	}

	var namePtr *byte
	status := s.runtime.apiFuncs.SessionGetInputName(s.ptr, uintptr(index), s.runtime.allocator.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get input name: %w", err)
	}

	name := cstrings.CStringToString(namePtr)

	// Free the allocated name
	s.runtime.allocator.free(unsafe.Pointer(namePtr))

	return name, nil
}

// getOutputName retrieves the output name from ONNX Runtime (internal use)
func (s *Session) getOutputName(index int) (string, error) {
	if s.ptr == 0 {
		return "", ErrSessionClosed
	}

	if s.runtime.allocator == nil {
		return "", errors.New("allocator not initialized")
	}

	var namePtr *byte
	status := s.runtime.apiFuncs.SessionGetOutputName(s.ptr, uintptr(index), s.runtime.allocator.ptr, &namePtr)
	if err := s.runtime.statusError(status); err != nil {
		return "", fmt.Errorf("failed to get output name: %w", err)
	}

	name := cstrings.CStringToString(namePtr)

	// Free the allocated name
	s.runtime.allocator.free(unsafe.Pointer(namePtr))

	return name, nil
}

// RunOption is a functional option for configuring inference execution.
type RunOption func(*runConfig)

type runConfig struct {
	outputNames []string
}

// WithOutputNames specifies which outputs to compute during inference.
// If not specified, all model outputs are computed.
func WithOutputNames(names ...string) RunOption {
	return func(c *runConfig) {
		c.outputNames = names
	}
}

// Run executes the model with the provided inputs and returns the computed outputs.
// The inputs parameter is a map from input name to tensor value.
func (s *Session) Run(ctx context.Context, inputs map[string]*Value, opts ...RunOption) (map[string]*Value, error) {
	if s.ptr == 0 {
		return nil, ErrSessionClosed
	}

	config := &runConfig{
		outputNames: s.outputNames, // default: all outputs
	}
	for _, opt := range opts {
		opt(config)
	}

	// Build input arrays from map using cached metadata
	inputNames := make([]string, 0, len(s.inputNames))
	inputValues := make([]*Value, 0, len(s.inputNames))

	for _, name := range s.inputNames {
		if value, ok := inputs[name]; ok {
			inputNames = append(inputNames, name)
			inputValues = append(inputValues, value)
		} else {
			inputNames = append(inputNames, "")
			inputValues = append(inputValues, nil)
		}
	}

	// Call the low-level run method
	outputValues, err := s.run(inputNames, inputValues, config.outputNames)
	if err != nil {
		return nil, err
	}

	// Convert output arrays to map
	outputs := make(map[string]*Value, len(outputValues))
	for i, value := range outputValues {
		outputs[config.outputNames[i]] = value
	}
	return outputs, nil
}

// run executes the model with the provided inputs and returns the computed outputs.
func (s *Session) run(inputNames []string, inputs []*Value, outputNames []string) ([]*Value, error) {
	if len(inputNames) != len(inputs) {
		return nil, fmt.Errorf("number of input names (%d) must match number of inputs (%d)", len(inputNames), len(inputs))
	}

	// Prepare input name pointers
	inputNamePtrs := make([]*byte, len(inputNames))
	for i, name := range inputNames {
		nameBytes := append([]byte(name), 0)
		inputNamePtrs[i] = &nameBytes[0]
	}

	// Prepare input value pointers
	inputValuePtrs := make([]api.OrtValue, len(inputs))
	for i, input := range inputs {
		if input != nil {
			inputValuePtrs[i] = input.ptr
		}
	}

	// Prepare output name pointers
	outputNamePtrs := make([]*byte, len(outputNames))
	for i, name := range outputNames {
		nameBytes := append([]byte(name), 0)
		outputNamePtrs[i] = &nameBytes[0]
	}

	// Prepare output value pointers
	outputValuePtrs := make([]api.OrtValue, len(outputNames))

	// Call Run
	status := s.runtime.apiFuncs.Run(
		s.ptr,
		0, // run options (NULL)
		&inputNamePtrs[0],
		&inputValuePtrs[0],
		uintptr(len(inputs)),
		&outputNamePtrs[0],
		uintptr(len(outputNames)),
		&outputValuePtrs[0],
	)
	if err := s.runtime.statusError(status); err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Wrap output values
	outputs := make([]*Value, len(outputValuePtrs))
	for i, ptr := range outputValuePtrs {
		outputs[i] = s.runtime.newValueFromPtr(ptr)
	}

	return outputs, nil
}

func (r *Runtime) configureIntraOpNumThreads(optsPtr api.OrtSessionOptions, options *SessionOptions) error {
	if options.IntraOpNumThreads <= 0 {
		return nil
	}
	status := r.apiFuncs.SetIntraOpNumThreads(optsPtr, int32(options.IntraOpNumThreads))
	if err := r.statusError(status); err != nil {
		return fmt.Errorf("failed to set intra-op num threads: %w", err)
	}
	return nil
}

// configureExecutionProviders configures execution providers for the session options.
func (r *Runtime) configureExecutionProviders(optsPtr api.OrtSessionOptions, options *SessionOptions) error {
	if len(options.ExecutionProviders) == 0 {
		return nil
	}

	for _, provider := range options.ExecutionProviders {
		providerNameBytes := append([]byte(provider), 0)

		// Call with no options (empty keys/values)
		status := r.apiFuncs.SessionOptionsAppendExecutionProvider(
			optsPtr,
			&providerNameBytes[0],
			nil, // no option keys
			nil, // no option values
			0,   // no options
		)
		if err := r.statusError(status); err != nil {
			return fmt.Errorf("failed to append execution provider %q: %w", provider, err)
		}
	}

	return nil
}

// Close releases the session and associated resources.
// It is safe to call Close multiple times.
func (s *Session) Close() {
	if s.ptr != 0 && s.runtime != nil && s.runtime.apiFuncs != nil {
		s.runtime.apiFuncs.ReleaseSession(s.ptr)
		s.ptr = 0
	}
}
