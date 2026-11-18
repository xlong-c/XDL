package templates

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCatalogAndValidateDataRoot(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := Load(
		filepath.Join(root, "dashboard", "templates", "catalog.json"),
		root,
		filepath.Join(root, "data"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, ok := catalog.Get("synthetic-classification"); !ok {
		t.Fatal("synthetic template is missing")
	}
	values, err := catalog.ValidateParameters("classification-cifar100", map[string]any{
		"runtime.experiment_name": "test-run",
		"runtime.data_dir":        "datasets/cifar",
		"trainer.max_epochs":      float64(3),
	})
	if err != nil {
		t.Fatal(err)
	}
	if values["runtime.data_dir"] != filepath.Join(root, "data", "datasets", "cifar") {
		t.Fatalf("unexpected data path: %v", values["runtime.data_dir"])
	}
}

func TestCatalogRejectsPathEscapeAndUnknownField(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := Load(
		filepath.Join(root, "dashboard", "templates", "catalog.json"),
		root,
		filepath.Join(root, "data"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := catalog.ValidateParameters("classification-cifar100", map[string]any{
		"runtime.data_dir": "../../etc",
	}); err == nil {
		t.Fatal("path escape was accepted")
	}
	if _, err := catalog.ValidateParameters("classification-cifar100", map[string]any{
		"model.target": "unsafe",
	}); err == nil {
		t.Fatal("unknown field was accepted")
	}
	if err := os.MkdirAll(filepath.Join(root, "data"), 0o755); err != nil {
		t.Fatal(err)
	}
}
