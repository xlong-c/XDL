package events

import (
	"testing"
	"time"
)

func TestHubReplaysRecentEventsAndAssignsIDs(t *testing.T) {
	hub := NewHub(32)
	first := hub.Publish(Event{RunID: "run-a", Type: "metrics", CreatedAt: time.Now()})
	second := hub.Publish(Event{RunID: "run-a", Type: "run_state", Reliable: true})
	if first.ID == 0 || second.ID <= first.ID {
		t.Fatalf("unexpected event IDs: %d, %d", first.ID, second.ID)
	}
	subscription := hub.Subscribe(first.ID, 8)
	defer subscription.Close()
	select {
	case event := <-subscription.C:
		if event.ID != second.ID {
			t.Fatalf("got event %d, want %d", event.ID, second.ID)
		}
	case <-time.After(time.Second):
		t.Fatal("expected replay event")
	}
}

func TestHubDropsSlowMetricsSubscriberWithoutBlockingPublisher(t *testing.T) {
	hub := NewHub(16)
	subscription := hub.Subscribe(0, 8)
	defer subscription.Close()
	started := time.Now()
	for i := 0; i < 1000; i++ {
		hub.Publish(Event{RunID: "run-a", Type: "metrics"})
	}
	if time.Since(started) > time.Second {
		t.Fatal("slow subscriber blocked event publisher")
	}
}
