package genai

import (
	"fmt"
	"unsafe"

	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
)

// Generator handles token-by-token text generation.
type Generator struct {
	ptr     api.OgaGenerator
	runtime *Runtime
}

// Close releases resources associated with the generator.
func (g *Generator) Close() {
	if g.ptr != 0 {
		g.runtime.funcs.DestroyGenerator(g.ptr)
		g.ptr = 0
	}
}

// IsDone checks if generation is complete.
// Returns true when the end-of-sequence token has been generated or max length reached.
func (g *Generator) IsDone() bool {
	return g.runtime.funcs.GeneratorIsDone(g.ptr)
}

// AppendTokens adds tokens to the generator's input sequence.
// This is used to provide the initial prompt tokens.
func (g *Generator) AppendTokens(tokens []int32) error {
	if len(tokens) == 0 {
		return nil
	}

	result := g.runtime.funcs.GeneratorAppendTokens(g.ptr, &tokens[0], uintptr(len(tokens)))
	if err := resultError(g.runtime.funcs, result); err != nil {
		return fmt.Errorf("failed to append tokens: %w", err)
	}
	return nil
}

// GenerateNextToken generates the next token in the sequence.
func (g *Generator) GenerateNextToken() error {
	result := g.runtime.funcs.GeneratorGenerateNextToken(g.ptr)
	if err := resultError(g.runtime.funcs, result); err != nil {
		return fmt.Errorf("failed to generate next token: %w", err)
	}
	return nil
}

// GetSequence returns the current token sequence at the specified batch index.
// For single-sequence generation, use index 0.
func (g *Generator) GetSequence(index int) ([]int32, error) {
	count := g.runtime.funcs.GeneratorGetSequenceCount(g.ptr, uintptr(index))
	if count == 0 {
		return []int32{}, nil
	}

	dataPtr := g.runtime.funcs.GeneratorGetSequenceData(g.ptr, uintptr(index))
	if dataPtr == nil {
		return nil, fmt.Errorf("failed to get sequence data")
	}

	// Copy tokens to Go slice
	tokens := make([]int32, count)
	srcTokens := unsafe.Slice(dataPtr, count)
	copy(tokens, srcTokens)

	return tokens, nil
}

// GetNextTokens returns the most recently generated tokens for all sequences in the batch.
// The returned slice contains one token per sequence (length equals batch size).
// This is more efficient than GetLastToken for batch generation.
func (g *Generator) GetNextTokens() ([]int32, error) {
	var tokensPtr *int32
	var count uintptr

	result := g.runtime.funcs.GeneratorGetNextTokens(g.ptr, &tokensPtr, &count)
	if err := resultError(g.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to get next tokens: %w", err)
	}

	if count == 0 || tokensPtr == nil {
		return []int32{}, nil
	}

	// Copy tokens to Go slice (pointer is only valid until next OgaGenerator call)
	tokens := make([]int32, count)
	srcTokens := unsafe.Slice(tokensPtr, count)
	copy(tokens, srcTokens)

	return tokens, nil
}
