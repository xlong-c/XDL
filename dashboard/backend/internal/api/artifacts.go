package api

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/store"
)

func indexRunArtifacts(ctx context.Context, database *store.Store, run store.Run) error {
	root, err := filepath.Abs(run.OutputDir)
	if err != nil {
		return err
	}
	var latestCheckpoint string
	var latestCheckpointTime time.Time
	count := 0
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if path == root {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if count >= 1000 {
			return filepath.SkipAll
		}
		info, err := entry.Info()
		if err != nil {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			return nil
		}
		kind := "file"
		if entry.IsDir() {
			if !strings.HasPrefix(relative, "checkpoints"+string(filepath.Separator)) {
				return nil
			}
			if !containsCheckpoint(path) {
				return nil
			}
			kind = "checkpoint"
			if info.ModTime().After(latestCheckpointTime) {
				latestCheckpoint = path
				latestCheckpointTime = info.ModTime()
			}
			count++
			if err := database.RecordArtifact(ctx, store.Artifact{
				RunID: run.ID, Kind: kind, RelativePath: filepath.ToSlash(relative),
				CreatedAt: info.ModTime().UTC(), Metadata: map[string]any{"modified_at": info.ModTime().UTC().Format(time.RFC3339Nano)},
			}); err != nil {
				return err
			}
			return filepath.SkipDir
		}
		extension := strings.ToLower(filepath.Ext(path))
		if strings.Contains(relative, "previews"+string(filepath.Separator)) && (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".webp") {
			kind = "preview"
		} else if extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".webp" {
			kind = "image"
		} else {
			return nil
		}
		count++
		return database.RecordArtifact(ctx, store.Artifact{
			RunID: run.ID, Kind: kind, RelativePath: filepath.ToSlash(relative),
			CreatedAt: info.ModTime().UTC(),
			Metadata:  map[string]any{"size_bytes": info.Size(), "extension": extension},
		})
	})
	if err != nil {
		return err
	}
	if latestCheckpoint != "" {
		return database.SetCheckpointPath(ctx, run.ID, latestCheckpoint)
	}
	return nil
}

func containsCheckpoint(directory string) bool {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return false
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		switch strings.ToLower(entry.Name()) {
		case "checkpoint.pt", "model.safetensors":
			return true
		}
	}
	return false
}
