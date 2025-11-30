.PHONY: help test test-docker test-local clean generate generate-ort generate-genai download-ort download-genai setup-workspace lint

# ONNX Runtime version (can be overridden)
ONNXRUNTIME_VERSION ?= 1.23.0
# ONNX Runtime GenAI version (can be overridden)
ONNXRUNTIME_GENAI_VERSION ?= 0.11.0

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OS := linux
	LIB_EXT := so
	LIB_PREFIX := lib
endif
ifeq ($(UNAME_S),Darwin)
	OS := darwin
	LIB_EXT := dylib
	LIB_PREFIX := lib
endif
ifeq ($(OS),Windows_NT)
	OS := windows
	LIB_EXT := dll
	LIB_PREFIX :=
endif

# ONNX Runtime library path (can be overridden)
# Auto-constructed: ./libs/{version}/lib/{lib_prefix}onnxruntime.{ext}
ONNXRUNTIME_LIB_PATH ?= $(shell pwd)/libs/$(ONNXRUNTIME_VERSION)/lib/$(LIB_PREFIX)onnxruntime.$(LIB_EXT)

# Test flags (can be overridden)
# Usage: make test FLAGS="-v -run TestName"
FLAGS ?=

# Default target
.DEFAULT_GOAL := help

# Show help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  make test                              - Run all tests"
	@echo "  make test FLAGS=-v                     - Run tests with verbose output"
	@echo "  make test FLAGS=\"-v -run TestName\"     - Run specific test with verbose"
	@echo "  make test FLAGS=-short                 - Run only short tests"
	@echo "  make clean                             - Clean test cache"
	@echo "  make generate                          - Generate all API bindings"
	@echo "  make generate-ort                      - Generate ONNX Runtime API bindings"
	@echo "  make generate-genai                    - Generate GenAI API bindings"
	@echo "  make download-ort                      - Download ONNX Runtime library"
	@echo "  make download-genai                    - Download GenAI library"
	@echo "  make setup-workspace                   - Setup go.work for local development"
	@echo "  make lint                              - Lint all modules in workspace"
	@echo "  make help                              - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  ONNXRUNTIME_VERSION                    - ONNX Runtime version (default: $(ONNXRUNTIME_VERSION))"
	@echo "  ONNXRUNTIME_GENAI_VERSION              - GenAI version (default: $(ONNXRUNTIME_GENAI_VERSION))"
	@echo "  ONNXRUNTIME_LIB_PATH                   - Path to ONNX Runtime library"
	@echo "                                           (default: auto-detected based on OS and version)"
	@echo "  FLAGS                                  - Additional test flags"
	@echo ""
	@echo "Current configuration:"
	@echo "  OS: $(OS)"
	@echo "  ONNX Runtime version: $(ONNXRUNTIME_VERSION)"
	@echo "  GenAI version: $(ONNXRUNTIME_GENAI_VERSION)"
	@echo "  Library path: $(ONNXRUNTIME_LIB_PATH)"

# Run tests
test: test-docker

# Run tests in Docker
test-docker:
	@echo "Running tests in Docker..."
	docker-compose -f compose.yml run --rm dev go test $(FLAGS) ./...

# Run tests locally (requires libraries installed)
test-local:
	ONNXRUNTIME_LIB_PATH=$(ONNXRUNTIME_LIB_PATH) go test $(FLAGS) ./...

# Clean test cache
clean:
	go clean -testcache

# Module path for workspace-wide operations
MODULE_PATH := github.com/shota3506/onnxruntime-purego

# Setup go.work for local development
setup-workspace:
	go work init . ./examples/resnet ./examples/roberta-sentiment ./examples/yolov10 ./examples/genai/phi3 ./examples/genai/phi3.5-vision

# Lint all modules in workspace
lint:
	go vet $(MODULE_PATH)/...
	staticcheck $(MODULE_PATH)/...

# Download ONNX Runtime library
download-ort:
	@echo "Downloading ONNX Runtime $(ONNXRUNTIME_VERSION)..."
	./download.sh $(ONNXRUNTIME_VERSION)

# Download GenAI library
download-genai:
	@echo "Downloading GenAI $(ONNXRUNTIME_GENAI_VERSION)..."
	./download_genai.sh $(ONNXRUNTIME_GENAI_VERSION)

# Generate all API bindings
generate: generate-ort generate-genai

# Generate ONNX Runtime API bindings
# Extract API version from ONNXRUNTIME_VERSION (e.g., 1.23.0 -> 23)
API_VERSION := $(shell echo $(ONNXRUNTIME_VERSION) | cut -d. -f2)
# Extract GenAI version components (e.g., 0.11.0 -> 0_11)
GENAI_API_VERSION := $(shell echo $(ONNXRUNTIME_GENAI_VERSION) | sed 's/\([0-9]*\)\.\([0-9]*\)\..*/\1_\2/')

generate-ort:
	@echo "Generating ONNX Runtime API bindings for version $(ONNXRUNTIME_VERSION)..."
	go run tools/codegen/main.go \
		-version $(ONNXRUNTIME_VERSION) \
		-out onnxruntime/internal/api/v$(API_VERSION)
	@echo "Generated bindings in onnxruntime/internal/api/v$(API_VERSION)"

# Generate GenAI API bindings
generate-genai:
	@echo "Generating GenAI API bindings for version $(ONNXRUNTIME_GENAI_VERSION)..."
	@# Check if header file exists
	@if [ ! -f "libs/genai/$(ONNXRUNTIME_GENAI_VERSION)/include/ort_genai_c.h" ]; then \
		echo "Error: Header file not found. Run 'make download-genai' first."; \
		exit 1; \
	fi
	go run tools/codegen/genai/main.go \
		-header libs/genai/$(ONNXRUNTIME_GENAI_VERSION)/include/ort_genai_c.h \
		-out genai/internal/api \
		-package api
	@echo "Generated bindings in genai/internal/api"
