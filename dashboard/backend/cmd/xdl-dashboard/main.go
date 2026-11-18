package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/xlong-c/xdl-dashboard/internal/api"
	"github.com/xlong-c/xdl-dashboard/internal/config"
	"github.com/xlong-c/xdl-dashboard/internal/events"
	"github.com/xlong-c/xdl-dashboard/internal/queue"
	"github.com/xlong-c/xdl-dashboard/internal/runner"
	"github.com/xlong-c/xdl-dashboard/internal/store"
	"github.com/xlong-c/xdl-dashboard/internal/system"
	"github.com/xlong-c/xdl-dashboard/internal/templates"
	"github.com/xlong-c/xdl-dashboard/internal/workflow"
)

func main() {
	configPath := flag.String("config", "./dashboard/config.json", "dashboard service JSON config")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("load dashboard config: %v", err)
	}
	database, err := store.Open(cfg.DatabasePath)
	if err != nil {
		log.Fatalf("open dashboard database: %v", err)
	}
	defer database.Close()
	catalog, err := templates.Load(cfg.TemplateCatalog, cfg.ProjectRoot, cfg.DataRoot)
	if err != nil {
		log.Fatalf("load dashboard templates: %v", err)
	}
	workflowCatalog, err := workflow.LoadCatalog(cfg.WorkflowCatalog)
	if err != nil {
		log.Fatalf("load workflow node catalog: %v", err)
	}
	pythonExecutable, err := workflow.ResolvePythonExecutable(cfg.PythonExecutable)
	if err != nil {
		log.Fatalf("resolve python executable: %v", err)
	}
	checker := workflow.NewCompilerChecker(workflow.CompilerConfig{
		PythonExecutable: pythonExecutable,
		ProjectRoot:      cfg.ProjectRoot,
		DataRoot:         cfg.DataRoot,
		Timeout:          cfg.CompilerTimeout,
	})
	hub := events.NewHub(cfg.EventBuffer)
	gpuProvider := system.NewNVIDIAProvider("")
	jobRunner := runner.New(cfg, database, hub)
	manager := queue.New(cfg, database, catalog, workflowCatalog, jobRunner, hub, gpuProvider)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := manager.Recover(ctx); err != nil {
		log.Fatalf("recover dashboard runs: %v", err)
	}

	server := api.NewServer(cfg, database, manager, catalog, workflowCatalog, checker, hub)
	collector := system.NewCollector(cfg.SystemInterval, gpuProvider)
	server.StartSystemStream(ctx, collector, jobRunner.ActiveProcessIDs)
	server.SetReady(true)

	httpServer := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           server,
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       65 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
	}()

	log.Printf("XDL dashboard listening on http://%s", cfg.ListenAddr)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("dashboard server failed: %v", err)
	}
}
