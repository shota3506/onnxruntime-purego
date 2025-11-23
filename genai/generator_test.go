package genai

import (
	"testing"
)

func TestGeneratorClose(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	generator, err := model.NewGenerator(nil)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}

	// Close should succeed
	generator.Close()

	// Second close should also succeed (idempotent)
	generator.Close()
}

func TestGeneratorAppendTokens(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)

	generator, err := model.NewGenerator(nil)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}
	defer generator.Close()

	tokens := []int32{1, 2, 3, 4, 5}
	if err := generator.AppendTokens(tokens); err != nil {
		t.Fatalf("Failed to append tokens: %v", err)
	}
}

func TestGeneratorGetSequence(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	// Encode some text
	prompt := "Hello"
	tokens, err := tokenizer.Encode(prompt)
	if err != nil {
		t.Fatalf("Failed to encode text: %v", err)
	}

	params := GeneratorParams{
		"max_length": int(20),
	}
	generator, err := model.NewGenerator(params)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}
	defer generator.Close()

	// Append tokens
	if err := generator.AppendTokens(tokens); err != nil {
		t.Fatalf("Failed to append tokens: %v", err)
	}

	// Get sequence at index 0
	seq, err := generator.GetSequence(0)
	if err != nil {
		t.Fatalf("Failed to get sequence: %v", err)
	}

	t.Logf("Input tokens: %v", tokens)
	t.Logf("Sequence: %v", seq)
}

func TestGeneratorGetNextTokens(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	// Encode prompt
	prompt := "Hello"
	tokens, err := tokenizer.Encode(prompt)
	if err != nil {
		t.Fatalf("Failed to encode: %v", err)
	}

	params := GeneratorParams{
		"max_length": int(20),
	}
	generator, err := model.NewGenerator(params)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}
	defer generator.Close()

	// Append tokens
	if err := generator.AppendTokens(tokens); err != nil {
		t.Fatalf("Failed to append tokens: %v", err)
	}

	// Generate one token
	if err := generator.GenerateNextToken(); err != nil {
		t.Fatalf("Failed to generate token: %v", err)
	}

	// Get next tokens
	nextTokens, err := generator.GetNextTokens()
	if err != nil {
		t.Fatalf("Failed to get next tokens: %v", err)
	}

	t.Logf("Next tokens: %v", nextTokens)
}

func TestGeneratorSimpleGeneration(t *testing.T) {
	rt := newTestRuntime(t)
	model := newTestModel(t, rt)
	tokenizer := newTestTokenizer(t, model)

	// Encode prompt
	prompt := "Hello"
	tokens, err := tokenizer.Encode(prompt)
	if err != nil {
		t.Fatalf("Failed to encode prompt: %v", err)
	}

	params := GeneratorParams{
		"max_length": int(20),
	}
	generator, err := model.NewGenerator(params)
	if err != nil {
		t.Fatalf("Failed to create generator: %v", err)
	}
	defer generator.Close()

	if err := generator.AppendTokens(tokens); err != nil {
		t.Fatalf("Failed to append tokens: %v", err)
	}

	// Generate tokens one by one
	for i := 0; !generator.IsDone(); i++ {
		if err := generator.GenerateNextToken(); err != nil {
			t.Fatalf("Failed to generate next token at iteration %d: %v", i, err)
		}
	}

	// Get the generated sequence
	seq, err := generator.GetSequence(0)
	if err != nil {
		t.Fatalf("Failed to get sequence: %v", err)
	}

	output, err := tokenizer.Decode(seq)
	if err != nil {
		t.Fatalf("Failed to decode output: %v", err)
	}

	t.Logf("Prompt: %q", prompt)
	t.Logf("Input tokens: %v", tokens)
	t.Logf("Generated tokens: %v", seq)
	t.Logf("Output: %q", output)
}
