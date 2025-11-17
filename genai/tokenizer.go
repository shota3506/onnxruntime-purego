package genai

import (
	"fmt"
	"unsafe"

	"github.com/shota3506/onnxruntime-purego/genai/internal/api"
	"github.com/shota3506/onnxruntime-purego/internal/cstrings"
)

// Tokenizer handles text tokenization and detokenization.
type Tokenizer struct {
	ptr     api.OgaTokenizer
	runtime *Runtime
}

// Close releases resources associated with the tokenizer.
func (t *Tokenizer) Close() {
	if t.ptr != 0 {
		t.runtime.funcs.DestroyTokenizer(t.ptr)
		t.ptr = 0
	}
}

// Encode converts text to token IDs.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	var seqPtr api.OgaSequences
	result := t.runtime.funcs.CreateSequences(&seqPtr)
	if err := resultError(t.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to create sequences: %w", err)
	}
	defer t.runtime.funcs.DestroySequences(seqPtr)

	// Encode the text
	textBytes := stringToBytes(text)
	result = t.runtime.funcs.TokenizerEncode(t.ptr, &textBytes[0], seqPtr)
	if err := resultError(t.runtime.funcs, result); err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// Get the token count for the first sequence (index 0)
	count := t.runtime.funcs.SequencesGetSequenceCount(seqPtr, 0)
	if count == 0 {
		return []int32{}, nil
	}

	// Get the token data
	dataPtr := t.runtime.funcs.SequencesGetSequenceData(seqPtr, 0)
	if dataPtr == nil {
		return nil, fmt.Errorf("failed to get sequence data")
	}

	// Copy tokens to Go slice
	tokens := make([]int32, count)
	srcTokens := unsafe.Slice(dataPtr, count)
	copy(tokens, srcTokens)

	return tokens, nil
}

// Decode converts token IDs to text.
func (t *Tokenizer) Decode(tokens []int32) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}

	var outStringPtr *byte
	result := t.runtime.funcs.TokenizerDecode(
		t.ptr,
		&tokens[0],
		uintptr(len(tokens)),
		&outStringPtr,
	)
	if err := resultError(t.runtime.funcs, result); err != nil {
		return "", fmt.Errorf("failed to decode tokens: %w", err)
	}

	if outStringPtr == nil {
		return "", nil
	}

	text := cstrings.CStringToString(outStringPtr)

	t.runtime.funcs.DestroyString(outStringPtr)

	return text, nil
}
