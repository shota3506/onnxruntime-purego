#!/bin/bash
#
# ONNX Runtime GenAI Download Script
# Usage: ./download_genai.sh [VERSION]
# Example: ./download_genai.sh 0.11.0
#

set -e

VERSION="${1:-0.11.0}"
LIBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/libs/genai"
TARGET_DIR="${LIBS_DIR}/${VERSION}"

# Detect platform
detect_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)

    case "$os" in
        darwin)
            OS="osx"
            ;;
        linux)
            OS="linux"
            ;;
        *)
            echo "Error: Unsupported OS: $os"
            exit 1
            ;;
    esac

    case "$arch" in
        x86_64|amd64)
            ARCH="x64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            echo "Error: Unsupported architecture: $arch"
            exit 1
            ;;
    esac

    echo "Detected platform: ${OS}-${ARCH}"
}

# Build download URL
build_url() {
    local base_url="https://github.com/microsoft/onnxruntime-genai/releases/download/v${VERSION}"
    FILENAME="onnxruntime-genai-${VERSION}-${OS}-${ARCH}.tar.gz"
    DOWNLOAD_URL="${base_url}/${FILENAME}"

    echo "Download URL: ${DOWNLOAD_URL}"
}

main() {
    echo "=========================================="
    echo "ONNX Runtime GenAI Download (v${VERSION})"
    echo "=========================================="

    detect_platform
    build_url

    # Skip if already exists
    if [ -d "$TARGET_DIR" ]; then
        echo "Already downloaded: $TARGET_DIR"
        exit 0
    fi

    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    cd "$TEMP_DIR"

    # Download
    echo "Downloading..."
    if command -v curl &> /dev/null; then
        curl -L -o "$FILENAME" --progress-bar "$DOWNLOAD_URL"
    elif command -v wget &> /dev/null; then
        wget -q --show-progress "$DOWNLOAD_URL"
    else
        echo "Error: curl or wget is required"
        exit 1
    fi

    # Extract
    echo "Extracting..."
    tar xzf "$FILENAME"

    # Find and move extracted directory
    EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "onnxruntime-genai-*" | head -n 1)
    if [ -z "$EXTRACTED_DIR" ]; then
        echo "Error: Extracted directory not found"
        exit 1
    fi

    # Move to libs directory
    mkdir -p "$LIBS_DIR"
    mv "$EXTRACTED_DIR" "$TARGET_DIR"

    echo "=========================================="
    echo "Completed: $TARGET_DIR"
    echo "=========================================="

    # Display library files
    echo ""
    echo "Library files:"
    if [ -d "$TARGET_DIR/lib" ]; then
        ls -lh "$TARGET_DIR/lib/"
    else
        ls -lh "$TARGET_DIR/"
    fi

    # Show usage hint
    echo ""
    echo "Usage example:"
    if [ "$OS" = "osx" ]; then
        echo "  go run ./examples/genai/ -lib $TARGET_DIR/lib/libonnxruntime-genai.dylib -model <model_path>"
    else
        echo "  go run ./examples/genai/ -lib $TARGET_DIR/lib/libonnxruntime-genai.so -model <model_path>"
    fi
}

main
