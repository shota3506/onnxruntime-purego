package cstrings

import (
	"strconv"
	"testing"
)

func TestCStringToString(t *testing.T) {
	testCases := []struct {
		input    []byte
		expected string
	}{
		{[]byte("hello\x00"), "hello"},
		{[]byte("world\x00extra"), "world"},
		{[]byte("\x00"), ""},
		{[]byte("test string\x00"), "test string"},
		{[]byte{0}, ""},
		{[]byte{'a', 0}, "a"},
		{[]byte("this is a longer test string with spaces and numbers 12345\x00"), "this is a longer test string with spaces and numbers 12345"},
		{[]byte("test\t\n\r\x00"), "test\t\n\r"}, // Special characters
		{[]byte("Hello 世界\x00"), "Hello 世界"},     // UTF-8 characters
	}

	for i, tc := range testCases {
		t.Run("Case"+strconv.Itoa(i), func(t *testing.T) {
			result := CStringToString(&tc.input[0])
			if result != tc.expected {
				t.Errorf("cStringToString(%q) = %q, expected %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestCStringToStringWithNil(t *testing.T) {
	result := CStringToString(nil)
	if result != "" {
		t.Errorf("Expected empty string for nil pointer, got %q", result)
	}
}
