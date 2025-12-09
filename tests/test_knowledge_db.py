"""Unit tests for knowledge_db module."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from src.knowledge_db import KnowledgeDatabase, KnowledgeEntry


class TestKnowledgeEntry:
    """Test KnowledgeEntry dataclass."""

    def test_knowledge_entry_creation(self):
        """Test creating a knowledge entry."""
        data = np.array([1, 2, 3, 4])
        entry = KnowledgeEntry(
            id=None,
            category="pattern",
            data_type="vision",
            data=data,
            label="class_a",
            metadata={"source": "test"},
            timestamp="2025-01-01 12:00:00",
        )

        assert entry.category == "pattern"
        assert entry.data_type == "vision"
        assert np.array_equal(entry.data, data)
        assert entry.label == "class_a"
        assert entry.metadata["source"] == "test"

    def test_knowledge_entry_to_dict(self):
        """Test converting entry to dictionary."""
        data = np.array([1.0, 2.0, 3.0])
        entry = KnowledgeEntry(
            id=1,
            category="sequence",
            data_type="digital",
            data=data,
            label=5,
            metadata={"test": "value"},
            timestamp="2025-01-01 12:00:00",
        )

        entry_dict = entry.to_dict()

        assert isinstance(entry_dict, dict)
        assert entry_dict["id"] == 1
        assert entry_dict["category"] == "sequence"
        assert "data_blob" in entry_dict
        assert isinstance(entry_dict["data_blob"], bytes)

    def test_knowledge_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = np.array([5, 6, 7, 8])
        data_blob = pickle.dumps(data)

        entry_dict = {
            "id": 2,
            "category": "sensorimotor",
            "data_type": "motor",
            "data_blob": data_blob,
            "label": json.dumps("target_x"),
            "metadata": json.dumps({"key": "val"}),
            "timestamp": "2025-01-01 12:00:00",
        }

        entry = KnowledgeEntry.from_dict(entry_dict)

        assert entry.id == 2
        assert entry.category == "sensorimotor"
        assert np.array_equal(entry.data, data)
        assert entry.label == "target_x"

    def test_knowledge_entry_roundtrip(self):
        """Test converting to dict and back."""
        data = np.array([[1, 2], [3, 4]])
        original = KnowledgeEntry(
            id=5,
            category="pattern",
            data_type="vision",
            data=data,
            label=10,
            metadata={"test": "metadata"},
            timestamp="2025-01-01 12:00:00",
        )

        entry_dict = original.to_dict()
        restored = KnowledgeEntry.from_dict(entry_dict)

        assert restored.id == original.id
        assert restored.category == original.category
        assert np.array_equal(restored.data, original.data)
        assert restored.label == original.label

    def test_knowledge_entry_with_none_label(self):
        """Test entry with no label."""
        data = np.array([1, 2, 3])
        entry = KnowledgeEntry(
            id=None,
            category="unlabeled",
            data_type="vision",
            data=data,
            label=None,
            metadata={},
            timestamp="2025-01-01 12:00:00",
        )

        entry_dict = entry.to_dict()
        assert entry_dict["label"] is None

        restored = KnowledgeEntry.from_dict(entry_dict)
        assert restored.label is None


class TestKnowledgeDatabase:
    """Test KnowledgeDatabase class."""

    def test_database_creation(self):
        """Test creating a knowledge database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            assert db.db_path == db_path
            assert db.conn is not None
            assert db_path.exists()

    def test_add_entry(self):
        """Test adding a single entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            data = np.array([1, 2, 3, 4])
            entry = KnowledgeEntry(
                id=None,
                category="pattern",
                data_type="vision",
                data=data,
                label="class_a",
                metadata={"source": "test"},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            entry_id = db.add_entry(entry)

            assert entry_id > 0
            assert isinstance(entry_id, int)

    def test_add_batch(self):
        """Test adding multiple entries at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            entries = []
            for i in range(5):
                data = np.array([i, i + 1, i + 2])
                entry = KnowledgeEntry(
                    id=None,
                    category="pattern",
                    data_type="vision",
                    data=data,
                    label=f"class_{i}",
                    metadata={"index": i},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                entries.append(entry)

            ids = db.add_batch(entries)

            assert len(ids) == 5
            assert all(id > 0 for id in ids)

    def test_query_all(self):
        """Test querying all entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add some entries
            for i in range(3):
                data = np.array([i])
                entry = KnowledgeEntry(
                    id=None,
                    category="test",
                    data_type="vision",
                    data=data,
                    label=i,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            # Query all
            results = db.query()

            assert len(results) == 3
            assert all(isinstance(e, KnowledgeEntry) for e in results)

    def test_query_by_category(self):
        """Test querying by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add entries with different categories
            for category in ["pattern", "sequence", "pattern"]:
                data = np.array([1, 2])
                entry = KnowledgeEntry(
                    id=None,
                    category=category,
                    data_type="vision",
                    data=data,
                    label=None,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            # Query pattern category
            results = db.query(category="pattern")

            assert len(results) == 2
            assert all(e.category == "pattern" for e in results)

    def test_query_by_data_type(self):
        """Test querying by data type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add entries with different data types
            for data_type in ["vision", "audition", "vision"]:
                data = np.array([1, 2])
                entry = KnowledgeEntry(
                    id=None,
                    category="pattern",
                    data_type=data_type,
                    data=data,
                    label=None,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            # Query vision data type
            results = db.query(data_type="vision")

            assert len(results) == 2
            assert all(e.data_type == "vision" for e in results)

    def test_query_with_limit(self):
        """Test querying with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add 10 entries
            for i in range(10):
                data = np.array([i])
                entry = KnowledgeEntry(
                    id=None,
                    category="test",
                    data_type="vision",
                    data=data,
                    label=i,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            # Query with limit
            results = db.query(limit=5)

            assert len(results) == 5

    def test_query_combined_filters(self):
        """Test querying with multiple filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add various entries
            configs = [
                ("pattern", "vision"),
                ("pattern", "audition"),
                ("sequence", "vision"),
                ("pattern", "vision"),
            ]

            for category, data_type in configs:
                data = np.array([1])
                entry = KnowledgeEntry(
                    id=None,
                    category=category,
                    data_type=data_type,
                    data=data,
                    label=None,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            # Query with both filters
            results = db.query(category="pattern", data_type="vision")

            assert len(results) == 2
            assert all(e.category == "pattern" and e.data_type == "vision" for e in results)

    def test_count(self):
        """Test counting entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Initially empty
            assert db.count() == 0

            # Add entries
            for i in range(7):
                data = np.array([i])
                entry = KnowledgeEntry(
                    id=None,
                    category="test",
                    data_type="vision",
                    data=data,
                    label=i,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            assert db.count() == 7

    def test_count_by_category(self):
        """Test counting by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add entries with different categories
            for i in range(3):
                data = np.array([i])
                entry = KnowledgeEntry(
                    id=None,
                    category="pattern",
                    data_type="vision",
                    data=data,
                    label=i,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            for i in range(2):
                data = np.array([i])
                entry = KnowledgeEntry(
                    id=None,
                    category="sequence",
                    data_type="vision",
                    data=data,
                    label=i,
                    metadata={},
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                db.add_entry(entry)

            assert db.count(category="pattern") == 3
            assert db.count(category="sequence") == 2

    def test_get_by_id(self):
        """Test getting entry by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add entry
            data = np.array([1, 2, 3])
            entry = KnowledgeEntry(
                id=None,
                category="test",
                data_type="vision",
                data=data,
                label="test_label",
                metadata={},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            entry_id = db.add_entry(entry)

            # Get by ID
            retrieved = db.get_by_id(entry_id)

            assert retrieved is not None
            assert retrieved.id == entry_id
            assert retrieved.category == "test"
            assert np.array_equal(retrieved.data, data)

    def test_get_by_id_not_found(self):
        """Test getting non-existent entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Try to get non-existent entry
            retrieved = db.get_by_id(999)

            assert retrieved is None

    def test_delete_entry(self):
        """Test deleting an entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Add entry
            data = np.array([1, 2, 3])
            entry = KnowledgeEntry(
                id=None,
                category="test",
                data_type="vision",
                data=data,
                label="test",
                metadata={},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            entry_id = db.add_entry(entry)

            assert db.count() == 1

            # Delete entry
            success = db.delete_entry(entry_id)

            assert success
            assert db.count() == 0

    def test_delete_entry_not_found(self):
        """Test deleting non-existent entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Try to delete non-existent entry
            success = db.delete_entry(999)

            assert not success

    def test_close_database(self):
        """Test closing database connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = KnowledgeDatabase(str(db_path))

            # Connection should be open
            assert db.conn is not None

            # Close connection
            db.close()

            # Connection should be closed (None or closed)
            # Note: sqlite3 connection doesn't become None, but we can't use it
            try:
                db.conn.cursor()  # This should fail if closed
                closed = False
            except:
                closed = True

            # If the close method exists and works, the connection should be unusable
            assert closed or db.conn is None

    def test_database_persistence(self):
        """Test that data persists across database instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create database and add entry
            db1 = KnowledgeDatabase(str(db_path))
            data = np.array([1, 2, 3])
            entry = KnowledgeEntry(
                id=None,
                category="test",
                data_type="vision",
                data=data,
                label="test_label",
                metadata={"key": "value"},
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            db1.add_entry(entry)
            db1.close()

            # Open same database in new instance
            db2 = KnowledgeDatabase(str(db_path))
            results = db2.query()

            assert len(results) == 1
            assert results[0].category == "test"
            assert results[0].label == "test_label"
            assert np.array_equal(results[0].data, data)
