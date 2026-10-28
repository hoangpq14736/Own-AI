import pytest
from core.models import Note
from core.memory import MemoryEngine, cosine_similarity


def test_note_model():
    note = Note(
        id=1,
        user_id="12345",
        text="Sample note #devops",
        tags="#devops",
        created_at="2026-10-28T00:00:00Z",
    )
    assert note.id == 1
    assert note["user_id"] == "12345"
    assert "text" in note
    assert note.get("tags") == "#devops"
    assert note.to_dict()["id"] == 1


def test_cosine_similarity():
    # Identical
    assert pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.0])) == 1.0
    # Orthogonal
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    # Opposite
    assert pytest.approx(cosine_similarity([1.0, 0.0], [-1.0, 0.0])) == -1.0
    # Zero vector
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    # Different lengths
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_notes_crud(tmp_path):
    db_file = tmp_path / "memory.db"
    mem = MemoryEngine(db_file)

    # Add notes
    nid1 = mem.add_note(user_id="u1", text="Note 1 for devops", tags="#devops", embedding=[0.5, 0.5])
    nid2 = mem.add_note(user_id="u1", text="Note 2 for general", tags="#general", embedding=[0.1, 0.9])
    assert nid1 > 0
    assert nid2 > nid1

    # List notes
    notes = mem.list_notes("u1")
    assert len(notes) == 2
    assert notes[0].id == nid2

    # Filter by tag
    devops_notes = mem.list_notes("u1", tag="#devops")
    assert len(devops_notes) == 1
    assert devops_notes[0].id == nid1

    # Get note
    note1 = mem.get_note(nid1, "u1")
    assert note1 is not None
    assert note1.text == "Note 1 for devops"

    # Search notes text
    found = mem.search_notes("u1", query="devops")
    assert len(found) == 1
    assert found[0].id == nid1

    # Search notes semantic vector
    found_sem = mem.search_notes("u1", query_embedding=[0.5, 0.5])
    assert len(found_sem) >= 1
    assert found_sem[0].id == nid1

    # Delete note
    deleted = mem.delete_note(nid1, "u1")
    assert deleted is True
    assert mem.get_note(nid1, "u1") is None
    assert len(mem.list_notes("u1")) == 1


def test_facts_semantic_search(tmp_path):
    db_file = tmp_path / "memory.db"
    mem = MemoryEngine(db_file)

    mem.add_fact("u1", "User prefers dark mode", embedding=[1.0, 0.0, 0.0])
    mem.add_fact("u1", "User is a senior Python developer", embedding=[0.0, 1.0, 0.0])
    mem.add_fact("u1", "User lives in Hanoi", embedding=[0.0, 0.0, 1.0])

    # Search for coding-related facts
    results = mem.search_facts_semantic("u1", query_embedding=[0.1, 0.9, 0.0], top_k=1)
    assert len(results) == 1
    assert "senior Python developer" in results[0]

    # Test build_memory_system_prompt with query_embedding
    prompt = mem.build_memory_system_prompt("u1", query_embedding=[0.1, 0.9, 0.0])
    assert "senior Python developer" in prompt
