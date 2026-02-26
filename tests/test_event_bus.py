from glitchlab.event_bus import EventBus, GlitchEvent


def test_event_bus_pub_sub():
    test_bus = EventBus()
    received_events: list[GlitchEvent] = []

    def dummy_subscriber(event: GlitchEvent):
        received_events.append(event)

    # Subscribe to the bus
    test_bus.subscribe(dummy_subscriber)
    
    # Emit an event
    test_bus.emit(
        event_type="TEST_EVENT", 
        agent_name="test_agent", 
        payload={"key": "value"}
    )

    # Verify the event was received and formatted correctly
    assert len(received_events) == 1
    
    event = received_events[0]
    assert event.event_type == "TEST_EVENT"
    assert event.agent_name == "test_agent"
    assert event.payload == {"key": "value"}
    
    # Verify auto-generated fields
    assert event.event_id is not None
    assert isinstance(event.event_id, str)
    assert event.timestamp is not None