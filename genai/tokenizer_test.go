package genai

import (
	"testing"
)

func TestTokenizerEncodeDecode(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	testCases := []string{
		"This is a test.",
		"She sells sea shells by the sea shore.",
		"Rats are awesome pets!",
		"Hello, world!",
	}

	for _, text := range testCases {
		t.Run(text, func(t *testing.T) {
			tokens, err := tokenizer.Encode(text)
			if err != nil {
				t.Fatalf("Failed to encode text: %v", err)
			}

			if len(tokens) == 0 {
				t.Fatal("Encoded tokens should not be empty")
			}

			decoded, err := tokenizer.Decode(tokens)
			if err != nil {
				t.Fatalf("Failed to decode tokens: %v", err)
			}

			// The decoded text might not exactly match the input due to tokenization,
			// but it should not be empty
			if decoded == "" {
				t.Fatal("Decoded text should not be empty")
			}

			t.Logf("Original: %q", text)
			t.Logf("Tokens: %v", tokens)
			t.Logf("Decoded: %q", decoded)
		})
	}
}

func TestTokenizerEncodeEmptyString(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	tokens, err := tokenizer.Encode("")
	if err != nil {
		t.Fatalf("Failed to encode empty string: %v", err)
	}

	// Empty string might result in empty tokens or special tokens
	t.Logf("Empty string tokens: %v", tokens)
}

func TestTokenizerDecodeEmptyTokens(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	decoded, err := tokenizer.Decode([]int32{})
	if err != nil {
		t.Fatalf("Failed to decode empty tokens: %v", err)
	}

	if decoded != "" {
		t.Fatalf("Expected empty string, got %q", decoded)
	}
}

func TestTokenizerClose(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	tokenizer, err := model.NewTokenizer()
	if err != nil {
		t.Fatalf("Failed to create tokenizer: %v", err)
	}

	// Close should succeed
	tokenizer.Close()

	// Second close should also succeed (idempotent)
	tokenizer.Close()
}
