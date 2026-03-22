from __future__ import annotations

import threading

from tardigrade._event_bus import EventBus


def test_event_bus_publish_and_poll_returns_events_in_order() -> None:
    bus = EventBus()
    events = [{"event": f"event_{index}"} for index in range(5)]

    for event in events:
        bus.publish(event)

    assert bus.poll() == events


def test_event_bus_poll_drains_queue() -> None:
    bus = EventBus()
    for index in range(3):
        bus.publish({"event": f"event_{index}"})

    assert len(bus.poll()) == 3
    assert bus.poll() == []


def test_event_bus_respects_max_events_when_polling() -> None:
    bus = EventBus()
    for index in range(200):
        bus.publish({"event": f"event_{index}"})

    first_batch = bus.poll(max_events=50)
    second_batch = bus.poll(max_events=50)

    assert len(first_batch) == 50
    assert len(second_batch) == 50
    assert first_batch[0]["event"] == "event_0"
    assert second_batch[0]["event"] == "event_50"


def test_event_bus_drops_events_when_queue_is_full() -> None:
    bus = EventBus(maxsize=5)
    for index in range(10):
        bus.publish({"event": f"event_{index}"})

    assert bus.poll() == [{"event": f"event_{index}"} for index in range(5)]


def test_event_bus_singleton_can_be_reset() -> None:
    first = EventBus.get()
    second = EventBus.get()

    EventBus.reset()
    third = EventBus.get()

    assert first is second
    assert third is not first


def test_event_bus_publish_is_thread_safe() -> None:
    bus = EventBus()

    def publish_batch(offset: int) -> None:
        for index in range(100):
            bus.publish({"event": f"event_{offset + index}"})

    threads = [
        threading.Thread(target=publish_batch, args=(batch * 100,))
        for batch in range(10)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = []
    while True:
        batch = bus.poll(max_events=200)
        if not batch:
            break
        events.extend(batch)

    assert len(events) == 1000
