package main

import (
	"bufio"
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"text/template"
)

//go:embed templates/api.go.tmpl
var apiTemplate string

//go:embed templates/funcs.go.tmpl
var funcsTemplate string

// CType represents a parsed C type
type CType struct {
	BaseType  string
	IsPointer int // number of pointer levels
	IsConst   bool
}

// Param represents a function parameter
type Param struct {
	Name   string
	CType  CType
	GoType string
}

// Function represents a parsed C function declaration
type Function struct {
	Name         string
	ReturnType   CType
	GoReturnType string
	Params       []Param
	GoName       string
}

// OpaqueType represents an opaque C type
type OpaqueType struct {
	CName  string
	GoName string
}

// GeneratorConfig holds the configuration for code generation
type GeneratorConfig struct {
	HeaderPath  string
	PackageName string
	OpaqueTypes []OpaqueType
	Functions   []Function
}

var (
	// Pattern to match function declarations
	// OGA_EXPORT <return_type> OGA_API_CALL <name>(<params>);
	funcPattern = regexp.MustCompile(`OGA_EXPORT\s+(.+?)\s+OGA_API_CALL\s+(\w+)\s*\(([^)]*)\)\s*;`)

	// Pattern to match typedef struct declarations
	// Note: Go regexp doesn't support backreferences, so we'll match all and filter
	typedefPattern = regexp.MustCompile(`typedef\s+struct\s+(\w+)\s+(\w+)\s*;`)
)

func main() {
	headerPath := flag.String("header", "", "Path to ort_genai_c.h header file")
	outDir := flag.String("out", "", "Output directory for generated code")
	packageName := flag.String("package", "", "Package name for generated code (default: derived from output directory)")
	flag.Parse()

	if *headerPath == "" || *outDir == "" {
		log.Fatal("Both -header and -out flags are required")
	}

	// Derive package name from output directory if not specified
	if *packageName == "" {
		*packageName = filepath.Base(*outDir)
	}

	log.Printf("Parsing header file: %s", *headerPath)
	log.Printf("Package name: %s", *packageName)

	file, err := os.Open(*headerPath)
	if err != nil {
		log.Fatalf("Failed to open header file: %v", err)
	}
	defer file.Close()

	// Read entire file content
	var content strings.Builder
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		content.WriteString(scanner.Text())
		content.WriteString("\n")
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("Failed to read header file: %v", err)
	}

	headerContent := content.String()

	// Parse opaque types
	opaqueTypes := parseOpaqueTypes(headerContent)
	log.Printf("Found %d opaque types", len(opaqueTypes))

	// Parse functions
	functions := parseFunctions(headerContent, opaqueTypes)
	log.Printf("Found %d functions", len(functions))

	// Create configuration
	config := GeneratorConfig{
		HeaderPath:  *headerPath,
		PackageName: *packageName,
		OpaqueTypes: opaqueTypes,
		Functions:   functions,
	}

	// Create output directory
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Generate api.go
	if err := executeTemplate(filepath.Join(*outDir, "api.go"), apiTemplate, config); err != nil {
		log.Fatalf("Failed to generate api.go: %v", err)
	}
	log.Println("Generated api.go")

	// Generate funcs.go
	if err := executeTemplate(filepath.Join(*outDir, "funcs.go"), funcsTemplate, config); err != nil {
		log.Fatalf("Failed to generate funcs.go: %v", err)
	}
	log.Println("Generated funcs.go")

	log.Println("Code generation completed successfully!")
}

func parseOpaqueTypes(content string) []OpaqueType {
	matches := typedefPattern.FindAllStringSubmatch(content, -1)

	var types []OpaqueType
	for _, match := range matches {
		structName := match[1]
		typeName := match[2]
		// Only include if struct name matches typedef name (opaque type pattern)
		if structName == typeName {
			goName := typeName // Keep same name for Go
			types = append(types, OpaqueType{
				CName:  typeName,
				GoName: goName,
			})
		}
	}

	return types
}

func parseFunctions(content string, opaqueTypes []OpaqueType) []Function {
	// Create a map for quick lookup of opaque types
	typeMap := make(map[string]bool)
	for _, t := range opaqueTypes {
		typeMap[t.CName] = true
	}

	matches := funcPattern.FindAllStringSubmatch(content, -1)

	var functions []Function
	for _, match := range matches {
		returnTypeStr := strings.TrimSpace(match[1])
		name := match[2]
		paramsStr := strings.TrimSpace(match[3])

		returnType := parseCType(returnTypeStr)
		goReturnType := cTypeToGoType(returnType, typeMap)

		var params []Param
		if paramsStr != "" && paramsStr != "void" {
			params = parseParams(paramsStr, typeMap)
		}

		// Generate Go-friendly name
		goName := strings.ReplaceAll(strings.TrimPrefix(name, "Oga"), "_", "")

		functions = append(functions, Function{
			Name:         name,
			ReturnType:   returnType,
			GoReturnType: goReturnType,
			Params:       params,
			GoName:       goName,
		})
	}

	return functions
}

func parseCType(typeStr string) CType {
	typeStr = strings.TrimSpace(typeStr)

	ctype := CType{}

	// Check for const
	if strings.HasPrefix(typeStr, "const ") {
		ctype.IsConst = true
		typeStr = strings.TrimPrefix(typeStr, "const ")
	}

	// Count pointer levels
	ctype.IsPointer = strings.Count(typeStr, "*")
	typeStr = strings.ReplaceAll(typeStr, "*", "")
	typeStr = strings.TrimSpace(typeStr)

	// Check for const after type
	if strings.HasSuffix(typeStr, " const") {
		ctype.IsConst = true
		typeStr = strings.TrimSuffix(typeStr, " const")
	}

	ctype.BaseType = typeStr

	return ctype
}

func parseParams(paramsStr string, typeMap map[string]bool) []Param {
	// Handle multi-line parameter declarations
	paramsStr = strings.ReplaceAll(paramsStr, "\n", " ")
	paramsStr = strings.Join(strings.Fields(paramsStr), " ")

	parts := strings.Split(paramsStr, ",")
	var params []Param

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		param := parseParam(part, typeMap)
		params = append(params, param)
	}

	return params
}

func parseParam(paramStr string, typeMap map[string]bool) Param {
	paramStr = strings.TrimSpace(paramStr)

	// Handle function pointer parameters (e.g., void (*callback)(...))
	if strings.Contains(paramStr, "(*") {
		// Extract the name between (* and )
		start := strings.Index(paramStr, "(*")
		end := strings.Index(paramStr[start:], ")")
		name := paramStr[start+2 : start+end]
		return Param{
			Name:   name,
			CType:  CType{BaseType: "void", IsPointer: 1},
			GoType: "uintptr",
		}
	}

	// Count pointers at the end
	pointerCount := strings.Count(paramStr, "*")
	paramStr = strings.ReplaceAll(paramStr, "*", " ")

	// Split into words
	words := strings.Fields(paramStr)
	if len(words) == 0 {
		return Param{Name: "arg", GoType: "uintptr"}
	}

	// Check if parameter has no name (just type)
	// This happens when we have just one word that's a type name (possibly with const)
	// e.g., "const OgaTokenizer" or "OgaTokenizer"
	isUnnamed := false
	var name string
	var typeWords []string
	isConst := false

	// Filter out const first
	var nonConstWords []string
	for _, w := range words {
		if w == "const" {
			isConst = true
		} else {
			nonConstWords = append(nonConstWords, w)
		}
	}

	if len(nonConstWords) == 1 {
		// Single word after removing const - this is either:
		// 1. An unnamed parameter with just a type name (e.g., "OgaTokenizer")
		// 2. A named parameter with just primitive type (e.g., "int32_t count" already split)
		word := nonConstWords[0]
		if typeMap[word] || isKnownType(word) {
			// It's a type, so this is an unnamed parameter
			isUnnamed = true
			typeWords = []string{word}
			name = "arg"
		} else {
			// It's a name, type must be inferred (shouldn't happen normally)
			name = word
			typeWords = []string{"int"}
		}
	} else if len(nonConstWords) > 1 {
		// Multiple words - last one is the name
		name = nonConstWords[len(nonConstWords)-1]
		typeWords = nonConstWords[:len(nonConstWords)-1]
	} else {
		name = "arg"
		typeWords = []string{"int"}
	}

	_ = isUnnamed // We track this but don't need to use it

	baseType := strings.Join(typeWords, " ")
	if baseType == "" {
		baseType = "int"
	}

	ctype := CType{
		BaseType:  baseType,
		IsPointer: pointerCount,
		IsConst:   isConst,
	}

	goType := cTypeToGoType(ctype, typeMap)

	return Param{
		Name:   name,
		CType:  ctype,
		GoType: goType,
	}
}

// isKnownType checks if a word is a known C type
func isKnownType(word string) bool {
	knownTypes := map[string]bool{
		"void": true, "bool": true, "int": true, "int32_t": true, "int64_t": true,
		"uint8_t": true, "uint16_t": true, "uint32_t": true, "uint64_t": true,
		"size_t": true, "char": true, "float": true, "double": true,
	}
	return knownTypes[word]
}

func cTypeToGoType(ctype CType, typeMap map[string]bool) string {
	base := ctype.BaseType

	// Map C base types to Go types
	goBase := ""
	switch base {
	case "void":
		if ctype.IsPointer > 0 {
			if ctype.IsPointer == 1 {
				return "uintptr"
			}
			return "*uintptr"
		}
		return ""
	case "bool":
		goBase = "bool"
	case "int32_t":
		goBase = "int32"
	case "int64_t":
		goBase = "int64"
	case "uint8_t":
		goBase = "uint8"
	case "uint32_t":
		goBase = "uint32"
	case "uint64_t":
		goBase = "uint64"
	case "size_t":
		goBase = "uintptr"
	case "int":
		goBase = "int32"
	case "double":
		goBase = "float64"
	case "float":
		goBase = "float32"
	case "char":
		if ctype.IsPointer >= 1 {
			if ctype.IsPointer == 1 {
				return "*byte"
			}
			if ctype.IsPointer == 2 {
				return "**byte"
			}
			return "uintptr"
		}
		goBase = "byte"
	default:
		// Check if it's an opaque type
		if typeMap[base] {
			// Opaque types are represented as uintptr in Go
			// When used as value (pointer level 0), return the type itself
			// When used as pointer (e.g., OgaModel*), return the type (since it's already uintptr)
			// When used as double pointer (e.g., OgaModel**), return *Type
			if ctype.IsPointer == 0 {
				return base
			}
			if ctype.IsPointer == 1 {
				// Single pointer to opaque type = the opaque type value
				return base
			}
			if ctype.IsPointer == 2 {
				// Double pointer = pointer to opaque type
				return "*" + base
			}
			return "uintptr"
		}
		// Unknown type, use uintptr
		return "uintptr"
	}

	// Add pointer levels
	result := goBase
	for i := 0; i < ctype.IsPointer; i++ {
		result = "*" + result
	}

	return result
}

func executeTemplate(path, tmplStr string, config GeneratorConfig) error {
	funcMap := template.FuncMap{
		"lower":     strings.ToLower,
		"hasPrefix": strings.HasPrefix,
	}

	tmpl, err := template.New("").Funcs(funcMap).Parse(tmplStr)
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, config); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	// Format the generated code
	formatted, err := format.Source(buf.Bytes())
	if err != nil {
		// If formatting fails, write unformatted code for debugging
		log.Printf("Warning: failed to format code: %v", err)
		formatted = buf.Bytes()
	}

	if err := os.WriteFile(path, formatted, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}
