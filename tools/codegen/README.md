# ONNX Runtime API Code Generator

Tools to automatically generate Go binding code from ONNX Runtime C API header files.

## Tools

### 1. ONNX Runtime Core API (`tools/codegen/main.go`)

Generates bindings for the core ONNX Runtime C API.

#### Usage

```bash
go run tools/codegen/main.go -version <VERSION> -out <OUTPUT_DIR>
```

Example:
```bash
go run tools/codegen/main.go -version 1.23.0 -out onnxruntime/internal/api/v23
```

#### Generated Files

**api.go**
- API version constant (`APIVersion`)
- `APIBase` structure (entry point)
- `API` structure (all function pointers, uintptr type)
- Function order matches the C header file

#### Header File Parsing

Recognizes the following 3 patterns:

1. **ORT_API2_STATUS macro**
   ```c
   ORT_API2_STATUS(CreateSession, ...)
   ```

2. **ORT_CLASS_RELEASE macro**
   ```c
   ORT_CLASS_RELEASE(Env);  // → ReleaseEnv
   ```

3. **Direct function pointer definition**
   ```c
   void(ORT_API_CALL* GetVersion)(...);
   ```

---

### 2. ONNX Runtime GenAI API (`tools/codegen/genai/main.go`)

Generates bindings for the ONNX Runtime GenAI C API.

#### Usage

```bash
go run tools/codegen/genai/main.go -header <HEADER_PATH> -out <OUTPUT_DIR>
```

Example:
```bash
# First, download the GenAI library to get the header file
./download_genai.sh

# Then generate the bindings
go run tools/codegen/genai/main.go \
  -header ./libs/genai/0.11.0/include/ort_genai_c.h \
  -out genai/internal/api
```

#### Generated Files

**api.go**
- Opaque pointer types (`OgaModel`, `OgaGenerator`, `OgaTokenizer`, etc.)
- All types are `uintptr` to represent C opaque pointers

**funcs.go**
- `Funcs` structure with function pointers for all GenAI C API functions
- `InitializeFuncs()` to load functions from shared library using purego
- `CheckResult()` helper for error handling
- `cStringToString()` helper for C string conversion

#### Header File Parsing

Parses the GenAI C header (`ort_genai_c.h`) to extract:

1. **Typedef struct declarations** (opaque types)
   ```c
   typedef struct OgaModel OgaModel;
   ```

2. **Function declarations**
   ```c
   OGA_EXPORT OgaResult* OGA_API_CALL OgaCreateModel(const char* config_path, OgaModel** out);
   ```

#### Type Mapping

| C Type | Go Type |
|--------|---------|
| `void` | (none) |
| `void*` | `uintptr` |
| `bool` | `bool` |
| `int32_t` | `int32` |
| `int64_t` | `int64` |
| `size_t` | `uintptr` |
| `double` | `float64` |
| `char*` | `*byte` |
| `char**` | `**byte` |
| `OgaModel*` | `OgaModel` (uintptr) |
| `OgaModel**` | `*OgaModel` |
| `OgaResult*` | `OgaResult` (uintptr) |
