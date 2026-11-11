package onnxruntime

import (
	"bytes"
	"os"
	"sync"
	"testing"
)

type logSink struct {
	mu    sync.Mutex
	lines []logLine
}

type logLine struct {
	severity LoggingLevel
	message  string
}

func (s *logSink) fn(severity LoggingLevel, category, logID, codeLocation, message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lines = append(s.lines, logLine{severity, message})
}

func (s *logSink) count(at LoggingLevel) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	n := 0
	for _, l := range s.lines {
		if l.severity == at && l.message != "" {
			n++
		}
	}
	return n
}

// An env created with a LogFunc hands it every line ONNX Runtime would
// otherwise print: a session created at verbose yields verbose or info
// lines, and SetLogLevel to warning stops them.
func TestNewEnvWithLoggerReceivesORTsLines(t *testing.T) {
	runtime := newTestRuntime(t)
	sink := &logSink{}
	env, err := runtime.NewEnvWithLogger("logger-test", LoggingLevelVerbose, sink.fn)
	if err != nil {
		t.Fatalf("NewEnvWithLogger: %v", err)
	}
	defer env.Close()

	modelData, err := os.ReadFile(testModelPath())
	if err != nil {
		t.Fatalf("read model: %v", err)
	}
	sess, err := runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
	if err != nil {
		t.Fatalf("NewSessionFromReader: %v", err)
	}
	sess.Close()
	if sink.count(LoggingLevelVerbose)+sink.count(LoggingLevelInfo) == 0 {
		t.Fatalf("no verbose or info line reached the LogFunc while creating a session: %+v", sink.lines)
	}

	if err := env.SetLogLevel(LoggingLevelWarning); err != nil {
		t.Fatalf("SetLogLevel: %v", err)
	}
	before := len(sink.lines)
	sess, err = runtime.NewSessionFromReader(env, bytes.NewReader(modelData), nil)
	if err != nil {
		t.Fatalf("NewSessionFromReader: %v", err)
	}
	sess.Close()
	sink.mu.Lock()
	defer sink.mu.Unlock()
	for _, l := range sink.lines[before:] {
		if l.severity < LoggingLevelWarning {
			t.Fatalf("a %d line reached the LogFunc after SetLogLevel(warning): %q", l.severity, l.message)
		}
	}
}

func TestNewEnvWithLoggerRefusesANilFunc(t *testing.T) {
	runtime := newTestRuntime(t)
	if _, err := runtime.NewEnvWithLogger("x", LoggingLevelWarning, nil); err == nil {
		t.Fatal("a nil LogFunc was accepted")
	}
}
