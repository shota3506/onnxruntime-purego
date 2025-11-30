package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/png"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/fogleman/gg"
	"github.com/nfnt/resize"
	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
)

var (
	modelPath  = flag.String("f", "", "path to ONNX model file (required)")
	confidence = flag.Float64("conf", 0.5, "confidence threshold")
	outputPath = flag.String("o", "", "output image path")
)

const (
	inputWidth  = 640
	inputHeight = 640
)

// Detection represents a detected object
type Detection struct {
	ClassID    int
	ClassName  string
	Confidence float32
	Box        BoundingBox
}

// BoundingBox represents a bounding box
type BoundingBox struct {
	X1, Y1, X2, Y2 float32
}

// COCO class names (80 classes)
var cocoClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush",
}

// preprocessImage loads and preprocesses an image for YOLOv10
func preprocessImage(imagePath string) ([]float32, image.Image, error) {
	// Open image file
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	// Decode image
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to decode image: %w", err)
	}

	// Resize to 640x640
	resized := resize.Resize(inputWidth, inputHeight, img, resize.Bilinear)

	// Convert to float32 tensor with shape [1, 3, 640, 640]
	// YOLOv10 expects normalized RGB values [0, 1]
	data := make([]float32, 1*3*inputHeight*inputWidth)

	for y := range inputHeight {
		for x := range inputWidth {
			r, g, b, _ := resized.At(x, y).RGBA()

			// RGBA() returns 16-bit color values (0-65535)
			// Convert to 0-1 range
			rVal := float32(r>>8) / 255.0
			gVal := float32(g>>8) / 255.0
			bVal := float32(b>>8) / 255.0

			// Store in CHW format (channels, height, width)
			idx := y*inputWidth + x
			data[0*inputHeight*inputWidth+idx] = rVal // R channel
			data[1*inputHeight*inputWidth+idx] = gVal // G channel
			data[2*inputHeight*inputWidth+idx] = bVal // B channel
		}
	}

	return data, img, nil
}

// parseYOLOv10Output parses YOLOv10 output format.
// YOLOv10 output shape: [1, 300, 6]
// where each detection is [xmin, ymin, xmax, ymax, score, class_id]
// YOLOv10 has NMS built-in, so no need for post-processing NMS.
func parseYOLOv10Output(output []float32, shape []int64, confThreshold float64, originalImg image.Image) []Detection {
	if len(shape) != 3 {
		fmt.Printf("Unexpected output shape: %v\n", shape)
		return nil
	}

	numDetections := int(shape[1])
	featuresPerDetection := int(shape[2])

	if featuresPerDetection != 6 {
		fmt.Printf("Expected 6 features per detection, got %d\n", featuresPerDetection)
		return nil
	}

	var detections []Detection

	origWidth := float32(originalImg.Bounds().Dx())
	origHeight := float32(originalImg.Bounds().Dy())
	scaleX := origWidth / inputWidth
	scaleY := origHeight / inputHeight

	for i := range numDetections {
		// YOLOv10 format: [xmin, ymin, xmax, ymax, score, class_id]
		offset := i * featuresPerDetection

		xmin := output[offset+0]
		ymin := output[offset+1]
		xmax := output[offset+2]
		ymax := output[offset+3]
		score := output[offset+4]
		classID := int(output[offset+5])

		// Filter by confidence threshold
		if float64(score) < confThreshold {
			continue
		}

		// Scale coordinates back to original image size
		x1 := xmin * scaleX
		y1 := ymin * scaleY
		x2 := xmax * scaleX
		y2 := ymax * scaleY

		className := "unknown"
		if classID >= 0 && classID < len(cocoClasses) {
			className = cocoClasses[classID]
		}

		detections = append(detections, Detection{
			ClassID:    classID,
			ClassName:  className,
			Confidence: score,
			Box: BoundingBox{
				X1: x1,
				Y1: y1,
				X2: x2,
				Y2: y2,
			},
		})
	}

	return detections
}

// drawDetections draws bounding boxes on the image
func drawDetections(img image.Image, detections []Detection, outputPath string) error {
	dc := gg.NewContextForImage(img)

	// Define colors for different classes (cycling through a palette)
	colors := []color.RGBA{
		{255, 0, 0, 255},   // Red
		{0, 255, 0, 255},   // Green
		{0, 0, 255, 255},   // Blue
		{255, 255, 0, 255}, // Yellow
		{255, 0, 255, 255}, // Magenta
		{0, 255, 255, 255}, // Cyan
		{255, 128, 0, 255}, // Orange
		{128, 0, 255, 255}, // Purple
		{0, 255, 128, 255}, // Spring Green
		{255, 0, 128, 255}, // Rose
	}

	// Draw each detection
	for _, det := range detections {
		boxColor := colors[det.ClassID%len(colors)]

		// Draw bounding box
		dc.SetColor(boxColor)
		dc.SetLineWidth(4)
		dc.DrawRectangle(
			float64(det.Box.X1),
			float64(det.Box.Y1),
			float64(det.Box.X2-det.Box.X1),
			float64(det.Box.Y2-det.Box.Y1),
		)
		dc.Stroke()
	}

	// Detect format based on extension and save
	ext := strings.ToLower(filepath.Ext(outputPath))
	switch ext {
	case ".jpg", ".jpeg":
		outFile, err := os.Create(outputPath)
		if err != nil {
			return fmt.Errorf("failed to create output file: %w", err)
		}
		defer outFile.Close()
		return jpeg.Encode(outFile, dc.Image(), &jpeg.Options{Quality: 90})
	case ".png":
		if err := dc.SavePNG(outputPath); err != nil {
			return fmt.Errorf("failed to save output image: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("unsupported output format: %s (supported: .png, .jpg, .jpeg)", ext)
	}
}

func run(ctx context.Context, modelPath, imagePath, outputFile string, confThreshold float64) error {
	// Load and preprocess image
	fmt.Printf("Loading image: %s\n", imagePath)
	inputData, originalImg, err := preprocessImage(imagePath)
	if err != nil {
		return fmt.Errorf("failed to preprocess image: %w", err)
	}

	fmt.Printf("Original image size: %dx%d\n", originalImg.Bounds().Dx(), originalImg.Bounds().Dy())

	// Initialize ONNX Runtime
	libraryPath := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if libraryPath == "" {
		return errors.New("ONNXRUNTIME_LIB_PATH environment variable not set")
	}

	// Create runtime
	runtime, err := ort.NewRuntime(libraryPath, 23)
	if err != nil {
		return fmt.Errorf("failed to create runtime: %w", err)
	}
	defer runtime.Close()

	// Create environment
	env, err := runtime.NewEnv("yolov10-example", ort.LoggingLevelWarning)
	if err != nil {
		return fmt.Errorf("failed to create environment: %w", err)
	}
	defer env.Close()

	// Open model file
	fmt.Printf("Loading model: %s\n", modelPath)
	modelFile, err := os.Open(modelPath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer modelFile.Close()

	// Create session
	session, err := runtime.NewSessionFromReader(env, modelFile, &ort.SessionOptions{
		IntraOpNumThreads: 1,
	})
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	defer session.Close()

	inputNames := session.InputNames()
	outputNames := session.OutputNames()

	fmt.Printf("Input names: %v\n", inputNames)
	fmt.Printf("Output names: %v\n", outputNames)

	// Create input tensor
	inputTensor, err := ort.NewTensorValue(runtime, inputData, []int64{1, 3, inputHeight, inputWidth})
	if err != nil {
		return fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Close()

	// Run inference
	outputs, err := session.Run(ctx, map[string]*ort.Value{
		inputNames[0]: inputTensor,
	})
	if err != nil {
		return fmt.Errorf("failed to run inference: %w", err)
	}

	// Get the output
	output := outputs[outputNames[0]]
	defer output.Close()

	outputData, shape, err := ort.GetTensorData[float32](output)
	if err != nil {
		return fmt.Errorf("failed to get output data: %w", err)
	}

	fmt.Printf("Output shape: %v\n", shape)

	// Parse detections
	detections := parseYOLOv10Output(outputData, shape, confThreshold, originalImg)
	fmt.Printf("Found %d detections\n", len(detections))

	// Display results
	fmt.Println("\nObject Detection")
	for i, det := range detections {
		fmt.Printf("%d. %s (%.2f%%)\n", i+1, det.ClassName, det.Confidence*100)
		fmt.Printf("   Box: [%.1f, %.1f, %.1f, %.1f]\n",
			det.Box.X1, det.Box.Y1, det.Box.X2, det.Box.Y2)
	}

	if len(detections) == 0 {
		fmt.Println("No objects detected. Try lowering the confidence threshold with -conf flag.")
		return nil
	}

	// Draw detections on image and save
	if outputFile != "" {
		fmt.Printf("\nDrawing bounding boxes on image...\n")
		if err := drawDetections(originalImg, detections, outputFile); err != nil {
			return fmt.Errorf("failed to draw detections: %w", err)
		}
		fmt.Printf("Output image saved to: %s\n", outputFile)
	}

	return nil
}

func main() {
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Error: -f flag is required\n\n")
		flag.Usage()
		os.Exit(1)
	}

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s -f <model_path> <image_path>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	imagePath := flag.Arg(0)

	ctx := context.Background()
	if err := run(ctx, *modelPath, imagePath, *outputPath, *confidence); err != nil {
		log.Fatalf("Error: %v", err)
	}
}
