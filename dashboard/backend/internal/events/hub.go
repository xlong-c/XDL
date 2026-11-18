package events

import (
	"sync"
	"time"
)

type Event struct {
	ID        int64
	RunID     string
	Type      string
	CreatedAt time.Time
	Payload   map[string]any
	Reliable  bool
}

type Subscription struct {
	C      <-chan Event
	cancel func()
}

func (s Subscription) Close() {
	if s.cancel != nil {
		s.cancel()
	}
}

type Hub struct {
	mu          sync.Mutex
	nextID      int64
	bufferSize  int
	recent      []Event
	subscribers map[int]chan Event
	nextSubID   int
}

func NewHub(bufferSize int) *Hub {
	if bufferSize < 16 {
		bufferSize = 16
	}
	return &Hub{
		bufferSize:  bufferSize,
		recent:      make([]Event, 0, bufferSize),
		subscribers: make(map[int]chan Event),
	}
}

func (h *Hub) Publish(event Event) Event {
	h.mu.Lock()
	defer h.mu.Unlock()
	if event.CreatedAt.IsZero() {
		event.CreatedAt = time.Now().UTC()
	}
	if event.ID == 0 {
		h.nextID++
		event.ID = h.nextID
	} else if event.ID > h.nextID {
		h.nextID = event.ID
	}
	h.recent = append(h.recent, event)
	if len(h.recent) > h.bufferSize {
		h.recent = h.recent[len(h.recent)-h.bufferSize:]
	}
	for id, subscriber := range h.subscribers {
		select {
		case subscriber <- event:
		default:
			if event.Reliable {
				delete(h.subscribers, id)
				close(subscriber)
			}
		}
	}
	return event
}

func (h *Hub) Subscribe(afterID int64, bufferSize int) Subscription {
	if bufferSize < 8 {
		bufferSize = 8
	}
	channel := make(chan Event, bufferSize)
	h.mu.Lock()
	id := h.nextSubID
	h.nextSubID++
	for _, event := range h.recent {
		if event.ID > afterID {
			select {
			case channel <- event:
			default:
				break
			}
		}
	}
	h.subscribers[id] = channel
	h.mu.Unlock()

	return Subscription{
		C: channel,
		cancel: func() {
			h.mu.Lock()
			if existing, ok := h.subscribers[id]; ok {
				delete(h.subscribers, id)
				close(existing)
			}
			h.mu.Unlock()
		},
	}
}

func (h *Hub) Recent(afterID int64) []Event {
	h.mu.Lock()
	defer h.mu.Unlock()
	result := make([]Event, 0)
	for _, event := range h.recent {
		if event.ID > afterID {
			result = append(result, event)
		}
	}
	return result
}
