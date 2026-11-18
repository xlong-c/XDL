package system

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type NVIDIAProvider struct {
	binary  string
	timeout time.Duration
}

func NewNVIDIAProvider(binary string) *NVIDIAProvider {
	if binary == "" {
		binary = findNvidiaSMI()
	}
	return &NVIDIAProvider{binary: binary, timeout: 1500 * time.Millisecond}
}

func (p *NVIDIAProvider) Available() bool {
	return p.binary != ""
}

func (p *NVIDIAProvider) Collect(ctx context.Context) ([]map[string]any, error) {
	if p.binary == "" {
		return nil, errors.New("nvidia-smi is unavailable")
	}
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	command := exec.CommandContext(
		ctx,
		p.binary,
		"--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits",
	)
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("query NVIDIA GPU status: %w", err)
	}
	reader := csv.NewReader(strings.NewReader(string(output)))
	reader.TrimLeadingSpace = true
	reader.FieldsPerRecord = 6
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi CSV: %w", err)
	}
	result := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		entry, err := parseGPUFields(row)
		if err != nil {
			return nil, err
		}
		result = append(result, entry)
	}
	return result, nil
}

func findNvidiaSMI() string {
	if path, err := exec.LookPath("nvidia-smi"); err == nil {
		return path
	}
	if runtime.GOOS == "windows" {
		programFiles := os.Getenv("ProgramFiles")
		candidate := filepath.Join(programFiles, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate
		}
	}
	return ""
}

func parseGPUFields(fields []string) (map[string]any, error) {
	if len(fields) != 6 {
		return nil, fmt.Errorf("expected 6 GPU fields, got %d", len(fields))
	}
	index, err := strconv.Atoi(strings.TrimSpace(fields[0]))
	if err != nil {
		return nil, fmt.Errorf("parse GPU index: %w", err)
	}
	values := make([]float64, 5)
	for i := 1; i < len(fields); i++ {
		value, err := strconv.ParseFloat(strings.TrimSpace(fields[i]), 64)
		if err != nil {
			if strings.Contains(strings.ToLower(fields[i]), "n/a") {
				values[i-1] = -1
				continue
			}
			return nil, fmt.Errorf("parse GPU metric %q: %w", fields[i], err)
		}
		values[i-1] = value
	}
	return map[string]any{
		"index":            index,
		"utilization":      values[0],
		"memory_used_mib":  values[1],
		"memory_total_mib": values[2],
		"temperature_c":    values[3],
		"power_w":          values[4],
	}, nil
}

var _ GPUProvider = (*NVIDIAProvider)(nil)
