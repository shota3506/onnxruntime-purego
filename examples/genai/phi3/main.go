package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/shota3506/onnxruntime-purego/genai"
)

var (
	modelPath = flag.String("model", "", "path to model directory (required)")
	maxLength = flag.Int("max-length", 200, "maximum number of tokens to generate")
)

func run(modelPath, prompt string, maxLength int) error {
	libraryPath := os.Getenv("ONNXRUNTIME_GENAI_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_GENAI_LIB_PATH environment variable not set")
	}

	// Initialize runtime
	runtime, err := genai.NewRuntime(libraryPath)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	// Load model
	fmt.Printf("Loading model: %s\n", modelPath)
	model, err := runtime.NewModel(modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer model.Close()

	// Create tokenizer
	tokenizer, err := model.NewTokenizer()
	if err != nil {
		return fmt.Errorf("failed to create tokenizer: %w", err)
	}
	defer tokenizer.Close()

	// Encode prompt
	tokens, err := tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("failed to encode prompt: %w", err)
	}
	fmt.Printf("Prompt tokens: %d\n", len(tokens))

	// Create generator with parameters
	generator, err := model.NewGenerator(genai.GeneratorParams{
		"max_length": maxLength,
	})
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}
	defer generator.Close()

	// Append prompt tokens
	if err := generator.AppendTokens(tokens); err != nil {
		return fmt.Errorf("failed to append tokens: %w", err)
	}

	// Generate tokens
	fmt.Println("\n--- Generation Output ---")
	fmt.Print(prompt)

	for !generator.IsDone() {
		if err := generator.GenerateNextToken(); err != nil {
			return fmt.Errorf("failed to generate token: %w", err)
		}

		nextTokens, err := generator.GetNextTokens()
		if err != nil {
			return fmt.Errorf("failed to get next tokens: %w", err)
		}

		text, err := tokenizer.Decode([]int32{nextTokens[0]})
		if err != nil {
			return fmt.Errorf("failed to decode token: %w", err)
		}

		fmt.Print(text)
	}

	fmt.Println("\n--- End of Generation ---")
	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -model flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <model_path> <prompt>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	prompt := flag.Arg(0)

	if err := run(*modelPath, prompt, *maxLength); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
