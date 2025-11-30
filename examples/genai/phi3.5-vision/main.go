package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/shota3506/onnxruntime-purego/genai"
)

var (
	modelPath  = flag.String("model", "", "path to model directory (required)")
	imagePaths = flag.String("images", "", "comma-separated paths to image files (required)")
	maxLength  = flag.Int("max-length", 2048, "maximum number of tokens to generate")
)

func run(modelPath string, imagePathList []string, prompt string, maxLength int) error {
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
	model, err := runtime.NewModel(modelPath, nil)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	defer model.Close()

	// Create multimodal processor
	processor, err := model.NewMultiModalProcessor()
	if err != nil {
		return fmt.Errorf("failed to create multimodal processor: %w", err)
	}
	defer processor.Close()

	// Load images
	fmt.Printf("Loading %d image(s)...\n", len(imagePathList))
	images, err := runtime.LoadImages(imagePathList)
	if err != nil {
		return fmt.Errorf("failed to load images: %w", err)
	}
	defer images.Close()

	// Build prompt with image tags
	// For Phi-3.5-vision, use format: <|user|>\n<|image_1|>\n...<|image_N|>\n{text}<|end|>\n<|assistant|>\n
	var promptBuilder strings.Builder
	promptBuilder.WriteString("<|user|>\n")
	for i := range imagePathList {
		promptBuilder.WriteString(fmt.Sprintf("<|image_%d|>\n", i+1))
	}
	promptBuilder.WriteString(prompt)
	promptBuilder.WriteString("<|end|>\n<|assistant|>\n")
	fullPrompt := promptBuilder.String()

	fmt.Printf("Prompt: %s\n", fullPrompt)

	// Process images with prompt
	fmt.Println("Processing images...")
	inputs, err := processor.ProcessImages(fullPrompt, images)
	if err != nil {
		return fmt.Errorf("failed to process images: %w", err)
	}
	defer inputs.Close()

	// Create generator with parameters
	generator, err := model.NewGenerator(genai.GeneratorParams{
		"max_length": maxLength,
	})
	if err != nil {
		return fmt.Errorf("failed to create generator: %w", err)
	}
	defer generator.Close()

	// Set inputs from processor
	if err := generator.SetInputs(inputs); err != nil {
		return fmt.Errorf("failed to set inputs: %w", err)
	}

	// Generate tokens
	fmt.Println("\n--- Generation Output ---")
	for !generator.IsDone() {
		if err := generator.GenerateNextToken(); err != nil {
			return fmt.Errorf("failed to generate token: %w", err)
		}

		nextTokens, err := generator.GetNextTokens()
		if err != nil {
			return fmt.Errorf("failed to get next tokens: %w", err)
		}

		text, err := processor.Decode([]int32{nextTokens[0]})
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

	if *imagePaths == "" {
		fmt.Fprintf(os.Stderr, "Error: -images flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s -model <model_path> -images <image1.jpg[,image2.jpg,...]> <prompt>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	imagePathList := strings.Split(*imagePaths, ",")
	for i := range imagePathList {
		imagePathList[i] = strings.TrimSpace(imagePathList[i])
	}

	prompt := flag.Arg(0)

	if err := run(*modelPath, imagePathList, prompt, *maxLength); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
