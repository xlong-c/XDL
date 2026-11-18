package templates

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

type Field struct {
	Path     string   `yaml:"path" json:"path"`
	Type     string   `yaml:"type" json:"type"`
	Label    string   `yaml:"label" json:"label"`
	Default  any      `yaml:"default,omitempty" json:"default,omitempty"`
	Min      any      `yaml:"min,omitempty" json:"min,omitempty"`
	Max      any      `yaml:"max,omitempty" json:"max,omitempty"`
	Required bool     `yaml:"required,omitempty" json:"required,omitempty"`
	Options  []string `yaml:"options,omitempty" json:"options,omitempty"`
}

type Definition struct {
	Version      int     `yaml:"version" json:"version"`
	ID           string  `yaml:"id" json:"id"`
	Label        string  `yaml:"label" json:"label"`
	Summary      string  `yaml:"summary" json:"summary"`
	RequiresGPU  bool    `yaml:"-" json:"requires_gpu"`
	SourceConfig string  `yaml:"source_config" json:"-"`
	Fields       []Field `yaml:"fields" json:"fields"`
}

type CatalogItem struct {
	ID              string   `json:"id"`
	Definition      string   `json:"definition"`
	SourceConfig    string   `json:"source_config"`
	AllowedOverride []string `json:"allowed_overrides"`
	RequiresGPU     bool     `json:"requires_gpu"`
}

type Catalog struct {
	projectRoot string
	dataRoot    string
	definitions map[string]Definition
	items       map[string]CatalogItem
}

type catalogFile struct {
	Version   int           `json:"version"`
	Templates []CatalogItem `json:"templates"`
}

func Load(catalogPath, projectRoot, dataRoot string) (*Catalog, error) {
	data, err := os.ReadFile(catalogPath)
	if err != nil {
		return nil, fmt.Errorf("read template catalog: %w", err)
	}
	var file catalogFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("decode template catalog: %w", err)
	}
	catalog := &Catalog{
		projectRoot: projectRoot,
		dataRoot:    dataRoot,
		definitions: make(map[string]Definition),
		items:       make(map[string]CatalogItem),
	}
	for _, item := range file.Templates {
		if item.ID == "" || item.Definition == "" || item.SourceConfig == "" {
			return nil, errors.New("template catalog entries require id, definition, and source_config")
		}
		if _, exists := catalog.items[item.ID]; exists {
			return nil, fmt.Errorf("duplicate template id %q", item.ID)
		}
		definitionPath, err := safeProjectPath(filepath.Dir(filepath.Dir(catalogPath)), item.Definition)
		if err != nil {
			return nil, fmt.Errorf("template %s definition: %w", item.ID, err)
		}
		definitionData, err := os.ReadFile(definitionPath)
		if err != nil {
			return nil, fmt.Errorf("read template %s definition: %w", item.ID, err)
		}
		var definition Definition
		if err := yaml.Unmarshal(definitionData, &definition); err != nil {
			return nil, fmt.Errorf("decode template %s definition: %w", item.ID, err)
		}
		if definition.ID != item.ID || definition.Version != file.Version {
			return nil, fmt.Errorf("template definition %s id/version mismatch", item.ID)
		}
		definition.SourceConfig = item.SourceConfig
		definition.RequiresGPU = item.RequiresGPU
		if _, err := safeProjectPath(projectRoot, item.SourceConfig); err != nil {
			return nil, fmt.Errorf("template %s source config: %w", item.ID, err)
		}
		if err := validateDefinition(definition, item); err != nil {
			return nil, err
		}
		catalog.items[item.ID] = item
		catalog.definitions[item.ID] = definition
	}
	return catalog, nil
}

func (c *Catalog) List() []Definition {
	result := make([]Definition, 0, len(c.definitions))
	for _, definition := range c.definitions {
		result = append(result, definition)
	}
	return result
}

func (c *Catalog) Get(id string) (Definition, CatalogItem, bool) {
	definition, ok := c.definitions[id]
	return definition, c.items[id], ok
}

func (c *Catalog) SourcePath(id string) (string, error) {
	_, item, ok := c.Get(id)
	if !ok {
		return "", fmt.Errorf("unknown template %q", id)
	}
	return safeProjectPath(c.projectRoot, item.SourceConfig)
}

func (c *Catalog) ValidateParameters(id string, submitted map[string]any) (map[string]any, error) {
	definition, item, ok := c.Get(id)
	if !ok {
		return nil, fmt.Errorf("unknown template %q", id)
	}
	fields := make(map[string]Field, len(definition.Fields))
	for _, field := range definition.Fields {
		fields[field.Path] = field
	}
	for path := range submitted {
		if _, allowed := fields[path]; !allowed || !contains(item.AllowedOverride, path) {
			return nil, fmt.Errorf("field %q is not allowed for template %s", path, id)
		}
	}
	result := make(map[string]any, len(fields))
	for _, field := range definition.Fields {
		value, provided := submitted[field.Path]
		if !provided {
			value = field.Default
		}
		if value == nil {
			if field.Required {
				return nil, fmt.Errorf("field %q is required", field.Path)
			}
			continue
		}
		if err := validateField(field, value); err != nil {
			return nil, fmt.Errorf("field %s: %w", field.Path, err)
		}
		if field.Type == "integer" {
			number, _ := numericValue(value)
			value = int(number)
		} else if field.Type == "number" {
			number, _ := numericValue(value)
			value = number
		}
		if field.Type == "data_directory" {
			path, err := c.resolveDataDirectory(value.(string))
			if err != nil {
				return nil, fmt.Errorf("field %s: %w", field.Path, err)
			}
			value = path
		}
		result[field.Path] = value
	}
	return result, nil
}

func (c *Catalog) resolveDataDirectory(value string) (string, error) {
	if filepath.IsAbs(value) {
		return "", errors.New("absolute paths are not allowed")
	}
	clean := filepath.Clean(value)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("path must stay inside the configured data root")
	}
	root, err := filepath.Abs(c.dataRoot)
	if err != nil {
		return "", err
	}
	target := filepath.Join(root, clean)
	relative, err := filepath.Rel(root, target)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", errors.New("path must stay inside the configured data root")
	}
	if resolved, err := filepath.EvalSymlinks(target); err == nil {
		resolvedRel, relErr := filepath.Rel(root, resolved)
		if relErr != nil || resolvedRel == ".." || strings.HasPrefix(resolvedRel, ".."+string(filepath.Separator)) {
			return "", errors.New("symlink path escapes the configured data root")
		}
	}
	return target, nil
}

func safeProjectPath(root, path string) (string, error) {
	if filepath.IsAbs(path) {
		return "", errors.New("absolute project paths are not allowed")
	}
	base, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	full := filepath.Join(base, filepath.Clean(path))
	rel, err := filepath.Rel(base, full)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("path escapes project root")
	}
	return full, nil
}

func validateDefinition(definition Definition, item CatalogItem) error {
	allowed := make(map[string]bool, len(item.AllowedOverride))
	for _, path := range item.AllowedOverride {
		allowed[path] = true
	}
	seen := make(map[string]bool, len(definition.Fields))
	for _, field := range definition.Fields {
		if field.Path == "" || !allowed[field.Path] {
			return fmt.Errorf("template %s has unapproved field %q", item.ID, field.Path)
		}
		if seen[field.Path] {
			return fmt.Errorf("template %s repeats field %q", item.ID, field.Path)
		}
		seen[field.Path] = true
	}
	return nil
}

func validateField(field Field, value any) error {
	switch field.Type {
	case "string", "data_directory":
		if _, ok := value.(string); !ok {
			return errors.New("must be a string")
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return errors.New("must be a boolean")
		}
	case "integer":
		number, ok := numericValue(value)
		if !ok || math.Trunc(number) != number {
			return errors.New("must be an integer")
		}
		if err := validateRange(field, number); err != nil {
			return err
		}
	case "number":
		number, ok := numericValue(value)
		if !ok || math.IsNaN(number) || math.IsInf(number, 0) {
			return errors.New("must be a finite number")
		}
		if err := validateRange(field, number); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unsupported field type %q", field.Type)
	}
	return nil
}

func validateRange(field Field, value float64) error {
	if field.Min != nil {
		minimum, ok := numericValue(field.Min)
		if !ok || value < minimum {
			return fmt.Errorf("must be >= %v", field.Min)
		}
	}
	if field.Max != nil {
		maximum, ok := numericValue(field.Max)
		if !ok || value > maximum {
			return fmt.Errorf("must be <= %v", field.Max)
		}
	}
	return nil
}

func numericValue(value any) (float64, bool) {
	switch number := value.(type) {
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case float64:
		return number, true
	default:
		return 0, false
	}
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
