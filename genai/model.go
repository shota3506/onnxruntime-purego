package genai

import (
	"fmt"

	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
)

// Model represents a loaded generative AI model.
type Model struct {
	ptr     api.OgaModel
	runtime *Runtime
}

// Close releases resources associated with the model.
func (m *Model) Close() {
	if m.ptr != 0 {
		m.runtime.funcs.DestroyModel(m.ptr)
		m.ptr = 0
	}
}

// NewTokenizer creates a tokenizer for this model.
func (m *Model) NewTokenizer() (*Tokenizer, error) {
	var tokenizerPtr api.OgaTokenizer
	result := m.runtime.funcs.CreateTokenizer(m.ptr, &tokenizerPtr)
	if err := resultError(m.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	return &Tokenizer{
		ptr:     tokenizerPtr,
		runtime: m.runtime,
	}, nil
}

// GeneratorParams holds parameters for text generation.
// Keys are parameter names (e.g., "max_length", "temperature", "do_sample").
// Values can be:
//   - int, int32, int64, float32, float64: for numeric parameters
//   - bool: for boolean parameters
//
// Common numeric parameters:
//   - "max_length": Maximum number of tokens to generate
//   - "min_length": Minimum number of tokens to generate
//   - "top_k": Number of highest probability tokens to keep
//   - "top_p": Cumulative probability threshold for nucleus sampling
//   - "temperature": Sampling temperature (higher = more random)
//   - "repetition_penalty": Penalty for repeating tokens
//   - "num_beams": Number of beams for beam search
//   - "num_return_sequences": Number of sequences to return
//   - "length_penalty": Length penalty for beam search
//   - "no_repeat_ngram_size": N-gram size for repetition penalty
//   - "random_seed": Random seed for sampling
//   - "batch_size": Batch size for generation
//
// Common boolean parameters:
//   - "do_sample": Whether to use sampling (true) or greedy decoding (false)
//   - "early_stopping": Stop generation when end token is found
//   - "past_present_share_buffer": Memory optimization for KV cache
type GeneratorParams map[string]any

// NewGenerator creates a generator with the specified parameters.
// If params is nil, default parameters will be used.
func (m *Model) NewGenerator(params GeneratorParams) (*Generator, error) {
	// Create C generator params
	var cParamsPtr api.OgaGeneratorParams
	result := m.runtime.funcs.CreateGeneratorParams(m.ptr, &cParamsPtr)
	if err := resultError(m.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create generator params: %w", err)
	}
	defer m.runtime.funcs.DestroyGeneratorParams(cParamsPtr)

	// Apply Go params to C params
	if len(params) > 0 {
		if err := m.applyGeneratorParams(cParamsPtr, params); err != nil {
			return nil, err
		}
	}

	// Create generator
	var generatorPtr api.OgaGenerator
	result = m.runtime.funcs.CreateGenerator(m.ptr, cParamsPtr, &generatorPtr)
	if err := resultError(m.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create generator: %w", err)
	}

	return &Generator{
		ptr:     generatorPtr,
		runtime: m.runtime,
	}, nil
}

// applyGeneratorParams applies Go GeneratorParams to C OgaGeneratorParams.
func (m *Model) applyGeneratorParams(cParams api.OgaGeneratorParams, params GeneratorParams) error {
	for name, value := range params {
		nameBytes := stringToBytes(name)
		switch v := value.(type) {
		case bool:
			result := m.runtime.funcs.GeneratorParamsSetSearchBool(cParams, &nameBytes[0], v)
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		case int:
			result := m.runtime.funcs.GeneratorParamsSetSearchNumber(cParams, &nameBytes[0], float64(v))
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		case int32:
			result := m.runtime.funcs.GeneratorParamsSetSearchNumber(cParams, &nameBytes[0], float64(v))
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		case int64:
			result := m.runtime.funcs.GeneratorParamsSetSearchNumber(cParams, &nameBytes[0], float64(v))
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		case float32:
			result := m.runtime.funcs.GeneratorParamsSetSearchNumber(cParams, &nameBytes[0], float64(v))
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		case float64:
			result := m.runtime.funcs.GeneratorParamsSetSearchNumber(cParams, &nameBytes[0], v)
			if err := resultError(m.runtime.funcs, result); err != nil {
				return fmt.Errorf("failed to set %q: %w", name, err)
			}
		default:
			return fmt.Errorf("unsupported parameter type for %q: %T", name, value)
		}
	}
	return nil
}
