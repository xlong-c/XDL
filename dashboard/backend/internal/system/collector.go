package system

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v4/cpu"
	"github.com/shirou/gopsutil/v4/disk"
	"github.com/shirou/gopsutil/v4/mem"
	"github.com/shirou/gopsutil/v4/net"
	"github.com/shirou/gopsutil/v4/process"
)

type GPUProvider interface {
	Collect(ctx context.Context) ([]map[string]any, error)
}

type Collector struct {
	interval time.Duration
	gpu      GPUProvider
	mu       sync.Mutex
	previous disk.IOCountersStat
	prevNet  net.IOCountersStat
	prevAt   time.Time
}

type Sample struct {
	CreatedAt time.Time
	Payload   map[string]any
}

func NewCollector(interval time.Duration, gpu GPUProvider) *Collector {
	if interval < 100*time.Millisecond {
		interval = 2 * time.Second
	}
	return &Collector{interval: interval, gpu: gpu}
}

func (c *Collector) Start(ctx context.Context, processIDs func() []int) <-chan Sample {
	output := make(chan Sample, 4)
	go func() {
		defer close(output)
		ticker := time.NewTicker(c.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				sample := c.Collect(ctx, processIDs())
				select {
				case output <- sample:
				default:
				}
			}
		}
	}()
	return output
}

func (c *Collector) Collect(ctx context.Context, processIDs []int) Sample {
	created := time.Now().UTC()
	payload := map[string]any{"timestamp": created.Format(time.RFC3339Nano)}
	if values, err := cpu.PercentWithContext(ctx, 0, false); err == nil && len(values) > 0 {
		payload["cpu_percent"] = values[0]
	}
	if value, err := mem.VirtualMemoryWithContext(ctx); err == nil {
		payload["memory"] = map[string]any{
			"total":     value.Total,
			"available": value.Available,
			"used":      value.Used,
			"percent":   value.UsedPercent,
		}
	}
	if value, err := disk.UsageWithContext(ctx, systemRoot()); err == nil {
		payload["disk"] = map[string]any{
			"total":   value.Total,
			"used":    value.Used,
			"free":    value.Free,
			"percent": value.UsedPercent,
		}
	}
	c.collectIO(ctx, created, payload)
	if c.gpu != nil {
		if values, err := c.gpu.Collect(ctx); err == nil {
			payload["gpus"] = values
		} else {
			payload["gpu_status"] = map[string]any{"available": false, "error": err.Error()}
		}
	}
	payload["processes"] = collectProcesses(ctx, processIDs)
	return Sample{CreatedAt: created, Payload: payload}
}

func (c *Collector) collectIO(ctx context.Context, now time.Time, payload map[string]any) {
	diskCounters, diskErr := disk.IOCountersWithContext(ctx)
	netCounters, netErr := net.IOCountersWithContext(ctx, false)
	c.mu.Lock()
	defer c.mu.Unlock()
	if diskErr == nil {
		var current disk.IOCountersStat
		for _, value := range diskCounters {
			current.ReadBytes += value.ReadBytes
			current.WriteBytes += value.WriteBytes
		}
		if !c.prevAt.IsZero() {
			seconds := now.Sub(c.prevAt).Seconds()
			if seconds > 0 && current.ReadBytes >= c.previous.ReadBytes && current.WriteBytes >= c.previous.WriteBytes {
				payload["disk_io"] = map[string]any{
					"read_bytes_per_sec":  float64(current.ReadBytes-c.previous.ReadBytes) / seconds,
					"write_bytes_per_sec": float64(current.WriteBytes-c.previous.WriteBytes) / seconds,
				}
			}
		}
		c.previous = current
	}
	if netErr == nil && len(netCounters) > 0 {
		current := netCounters[0]
		if !c.prevAt.IsZero() {
			seconds := now.Sub(c.prevAt).Seconds()
			if seconds > 0 && current.BytesSent >= c.prevNet.BytesSent && current.BytesRecv >= c.prevNet.BytesRecv {
				payload["network"] = map[string]any{
					"bytes_sent_per_sec": float64(current.BytesSent-c.prevNet.BytesSent) / seconds,
					"bytes_recv_per_sec": float64(current.BytesRecv-c.prevNet.BytesRecv) / seconds,
				}
			}
		}
		c.prevNet = current
	}
	c.prevAt = now
}

func systemRoot() string {
	if runtime.GOOS != "windows" {
		return string(filepath.Separator)
	}
	workingDirectory, err := os.Getwd()
	if err != nil {
		return `C:\`
	}
	volume := filepath.VolumeName(workingDirectory)
	if volume == "" {
		return `C:\`
	}
	return volume + string(filepath.Separator)
}

func collectProcesses(ctx context.Context, roots []int) map[string]any {
	result := map[string]any{"count": 0, "rss_bytes": uint64(0), "cpu_percent": float64(0)}
	processes, err := process.ProcessesWithContext(ctx)
	if err != nil {
		return result
	}
	byPID := make(map[int32]*process.Process, len(processes))
	children := make(map[int32][]int32)
	for _, value := range processes {
		pid := value.Pid
		byPID[pid] = value
		if parent, err := value.PpidWithContext(ctx); err == nil {
			children[parent] = append(children[parent], pid)
		}
	}
	seen := make(map[int32]bool)
	queue := make([]int32, 0, len(roots))
	for _, root := range roots {
		if root > 0 {
			queue = append(queue, int32(root))
		}
	}
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if seen[pid] {
			continue
		}
		seen[pid] = true
		queue = append(queue, children[pid]...)
		value := byPID[pid]
		if value == nil {
			continue
		}
		if memory, err := value.MemoryInfoWithContext(ctx); err == nil && memory != nil {
			result["rss_bytes"] = result["rss_bytes"].(uint64) + memory.RSS
		}
		if percent, err := value.PercentWithContext(ctx, 0); err == nil {
			result["cpu_percent"] = result["cpu_percent"].(float64) + percent
		}
		result["count"] = result["count"].(int) + 1
	}
	return result
}
