import os
from datetime import datetime

# Adjust this import based on your actual file name
from nodes import export_events_pdf_node
from models import Event, EventCategory


async def test_export_events_pdf_node_no_events():
    """
    Test that if the state contains no events, the node returns an empty list
    for pdf_files and updates the current_step without crashing.
    """
    initial_state = {
        "user_query": "events in Florence",
        "target_location": "Florence",
        "events": [],
        "current_step": "db_saved_empty"
    }

    new_state = await export_events_pdf_node(initial_state)

    # Verify state updates
    assert new_state.get("pdf_files") == []
    assert new_state.get("current_step") == "pdf_exported"
    assert new_state.get("target_location") == "Florence"


async def test_export_events_pdf_node_creates_pdf(monkeypatch, tmp_path):
    """
    Test that the node successfully exports a PDF file to the disk
    when a valid Event object is provided.
    """
    # 1. SETUP: Change the working directory to a temporary folder
    # This ensures the "pdf_events" folder is created inside the pytest temp dir
    # instead of polluting your actual project repository.
    monkeypatch.chdir(tmp_path)

    # Create a mock event
    mock_event = Event(
        title="Test Hardgroove Party",
        category=EventCategory.CLUB,
        start_datetime=datetime(2026, 12, 31, 23, 0, 0),
        venue="Test Club",
        city="Florence",
        source_url="https://example.com/test",
        expected_reach=500
    )

    initial_state = {
        "events": [mock_event],
        "current_step": "db_saved"
    }

    # 2. EXECUTION: Run the node[cite: 1]
    new_state = await export_events_pdf_node(initial_state)

    # 3. VERIFICATION: Check the state and the filesystem
    pdf_files = new_state.get("pdf_files")

    # Check that exactly one file path was returned[cite: 1]
    assert len(pdf_files) == 1
    assert new_state.get("current_step") == "pdf_exported"

    created_filepath = pdf_files[0]

    # Verify the file was physically created on the disk[cite: 1]
    assert os.path.exists(created_filepath)

    # Verify it has the correct extension and naming convention[cite: 1]
    assert created_filepath.endswith(".pdf")
    assert "Test_Hardgroove_Party" in created_filepath