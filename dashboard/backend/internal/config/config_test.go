package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadUsesSafeDefaultsAndResolvesPaths(t *testing.T) {
	root := t.TempDir()
	configPath := filepath.Join(root, "config.json")
	if err := os.WriteFile(configPath, []byte(`{"project_root":"."}`), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ListenAddr != "127.0.0.1:8765" {
		t.Fatalf("unexpected listen address: %s", cfg.ListenAddr)
	}
	if cfg.ProjectRoot != root {
		t.Fatalf("project root was not resolved: %s", cfg.ProjectRoot)
	}
	if cfg.SystemInterval.String() != "2s" {
		t.Fatalf("unexpected interval: %s", cfg.SystemInterval)
	}
}

func TestLoadRejectsNonLoopbackListener(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(configPath, []byte(`{"listen_addr":"0.0.0.0:8765"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Load(configPath)
	if err == nil || !strings.Contains(err.Error(), "non-loopback") {
		t.Fatalf("expected non-loopback error, got %v", err)
	}
}

func TestLoopbackAddressSupportsIPv4AndIPv6(t *testing.T) {
	for _, address := range []string{"127.0.0.1:9123", "[::1]:9123", "localhost:9123"} {
		if !isLoopbackAddress(address) {
			t.Errorf("expected loopback address: %s", address)
		}
	}
	if isLoopbackAddress("192.168.1.2:9123") {
		t.Fatal("non-loopback address was accepted")
	}
}
