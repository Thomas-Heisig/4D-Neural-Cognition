"""Knowledge database system for 4D Neural Cognition.

This module provides a database for storing training data and knowledge
that the neural network can access and train on, even before it has
learned from direct experience.
"""

import base64
import json
import sqlite3
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge database."""

    id: Optional[int]
    category: str  # e.g., 'pattern', 'sequence', 'sensorimotor'
    data_type: str  # e.g., 'vision', 'audition', 'digital', 'motor'
    data: np.ndarray  # The actual data
    label: Optional[Any]  # Optional label/target
    metadata: Dict[str, Any]  # Additional metadata
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage.
        
        Uses numpy's native save format (NPY) encoded as base64 instead of pickle
        for better security. NPY format cannot execute arbitrary code.
        """
        # Use numpy's native binary format instead of pickle
        buffer = BytesIO()
        np.save(buffer, self.data, allow_pickle=False)  # Explicitly disable pickle
        data_blob = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "id": self.id,
            "category": self.category,
            "data_type": self.data_type,
            "data_blob": data_blob,
            "label": json.dumps(self.label) if self.label is not None else None,
            "metadata": json.dumps(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create from dictionary.
        
        Loads numpy arrays from NPY format (base64 encoded) instead of pickle
        for security reasons.
        """
        # Decode from base64 and load using numpy's native format
        data_blob = base64.b64decode(data["data_blob"])
        buffer = BytesIO(data_blob)
        data_array = np.load(buffer, allow_pickle=False)  # Explicitly disable pickle
        
        return cls(
            id=data["id"],
            category=data["category"],
            data_type=data["data_type"],
            data=data_array,
            label=json.loads(data["label"]) if data["label"] else None,
            metadata=json.loads(data["metadata"]),
            timestamp=data["timestamp"],
        )


class KnowledgeDatabase:
    """Database for storing and retrieving training knowledge."""

    def __init__(self, db_path: str = "knowledge.db"):
        """Initialize knowledge database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Create knowledge entries table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                data_type TEXT NOT NULL,
                data_blob BLOB NOT NULL,
                label TEXT,
                metadata TEXT,
                timestamp TEXT NOT NULL
            )
        """
        )

        # Create index for faster queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_category_type
            ON knowledge_entries(category, data_type)
        """
        )

        self.conn.commit()

    def add_entry(self, entry: KnowledgeEntry) -> int:
        """Add a knowledge entry to the database.

        Args:
            entry: Knowledge entry to add

        Returns:
            ID of the inserted entry
        """
        cursor = self.conn.cursor()
        entry_dict = entry.to_dict()

        cursor.execute(
            """
            INSERT INTO knowledge_entries
            (category, data_type, data_blob, label, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                entry_dict["category"],
                entry_dict["data_type"],
                entry_dict["data_blob"],
                entry_dict["label"],
                entry_dict["metadata"],
                entry_dict["timestamp"],
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def add_batch(self, entries: List[KnowledgeEntry]) -> List[int]:
        """Add multiple entries efficiently.

        Args:
            entries: List of knowledge entries

        Returns:
            List of inserted IDs
        """
        cursor = self.conn.cursor()
        ids = []

        for entry in entries:
            entry_dict = entry.to_dict()
            cursor.execute(
                """
                INSERT INTO knowledge_entries
                (category, data_type, data_blob, label, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    entry_dict["category"],
                    entry_dict["data_type"],
                    entry_dict["data_blob"],
                    entry_dict["label"],
                    entry_dict["metadata"],
                    entry_dict["timestamp"],
                ),
            )
            ids.append(cursor.lastrowid)

        self.conn.commit()
        return ids

    def query(
        self,
        category: Optional[str] = None,
        data_type: Optional[str] = None,
        limit: Optional[int] = None,
        random_sample: bool = False,
    ) -> List[KnowledgeEntry]:
        """Query knowledge entries.

        Args:
            category: Filter by category
            data_type: Filter by data type
            limit: Maximum number of results
            random_sample: If True, return random sample

        Returns:
            List of matching knowledge entries
        """
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT * FROM knowledge_entries WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)

        if random_sample:
            query += " ORDER BY RANDOM()"
        else:
            query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to KnowledgeEntry objects
        entries = []
        for row in rows:
            entry_dict = dict(row)
            entries.append(KnowledgeEntry.from_dict(entry_dict))

        return entries

    def get_by_id(self, entry_id: int) -> Optional[KnowledgeEntry]:
        """Get a specific entry by ID.

        Args:
            entry_id: ID of the entry

        Returns:
            Knowledge entry or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()

        if row:
            return KnowledgeEntry.from_dict(dict(row))
        return None

    def count(self, category: Optional[str] = None, data_type: Optional[str] = None) -> int:
        """Count entries matching criteria.

        Args:
            category: Filter by category
            data_type: Filter by data type

        Returns:
            Number of matching entries
        """
        cursor = self.conn.cursor()

        query = "SELECT COUNT(*) FROM knowledge_entries WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)

        cursor.execute(query, params)
        return cursor.fetchone()[0]

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry.

        Args:
            entry_id: ID of entry to delete

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "KnowledgeDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class KnowledgeBasedTrainer:
    """Trainer that uses knowledge database for pre-training and continued learning."""

    def __init__(self, simulation, knowledge_db: KnowledgeDatabase):
        """Initialize knowledge-based trainer.

        Args:
            simulation: Simulation instance to train
            knowledge_db: Knowledge database to use
        """
        self.simulation = simulation
        self.knowledge_db = knowledge_db

    def pretrain(
        self, category: str, num_samples: int = 100, steps_per_sample: int = 50, random_order: bool = True
    ) -> Dict[str, Any]:
        """Pre-train the network using knowledge from database.

        Args:
            category: Category of knowledge to train on
            num_samples: Number of samples to use
            steps_per_sample: Simulation steps per sample
            random_order: Whether to randomize sample order

        Returns:
            Training statistics
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        print(f"Pre-training on category: {category}")
        print(f"Loading {num_samples} samples from database...")

        # Query knowledge entries
        entries = self.knowledge_db.query(category=category, limit=num_samples, random_sample=random_order)

        if not entries:
            print(f"Warning: No entries found for category '{category}'")
            return {"samples_processed": 0, "error": "No data found"}

        print(f"Found {len(entries)} samples")

        # Train on each sample
        stats = {"samples_processed": 0, "total_steps": 0, "total_spikes": 0, "avg_activity": 0.0}

        for i, entry in enumerate(entries):
            if (i + 1) % 10 == 0:
                print(f"Processing sample {i + 1}/{len(entries)}...")

            # Feed data to appropriate sense
            feed_sense_input(self.simulation.model, entry.data_type, entry.data)

            # Run simulation steps
            sample_spikes = 0
            for _ in range(steps_per_sample):
                step_stats = self.simulation.step()
                sample_spikes += len(step_stats["spikes"])

            stats["samples_processed"] += 1
            stats["total_steps"] += steps_per_sample
            stats["total_spikes"] += sample_spikes

        # Calculate averages
        if stats["total_steps"] > 0:
            stats["avg_activity"] = stats["total_spikes"] / stats["total_steps"]

        print("\nPre-training completed:")
        print(f"  Samples: {stats['samples_processed']}")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Average activity: {stats['avg_activity']:.2f} spikes/step")

        return stats

    def train_with_fallback(
        self, current_data: np.ndarray, data_type: str, category: str, steps: int = 50, use_database: bool = True
    ) -> Dict[str, Any]:
        """Train on current data, with fallback to database if network is untrained.

        This implements the key requirement: if the network hasn't learned yet,
        it can still access and train on database knowledge.

        Args:
            current_data: Current input data
            data_type: Type of data (e.g., 'vision', 'digital')
            category: Category for database fallback
            steps: Steps to train
            use_database: Whether to use database fallback

        Returns:
            Training statistics
        """
        try:
            from .senses import feed_sense_input
        except ImportError:
            from senses import feed_sense_input

        # Feed current data
        feed_sense_input(self.simulation.model, data_type, current_data)

        # Run simulation
        current_spikes = 0
        for _ in range(steps):
            step_stats = self.simulation.step()
            current_spikes += len(step_stats["spikes"])

        # Check if network is responding
        avg_activity = current_spikes / steps

        stats = {"current_activity": avg_activity, "used_database": False, "database_samples": 0}

        # If activity is very low and database is available, use it
        if avg_activity < 0.1 and use_database:
            print("Low activity detected, accessing knowledge database...")

            # Get similar examples from database
            db_entries = self.knowledge_db.query(category=category, data_type=data_type, limit=10, random_sample=True)

            if db_entries:
                print(f"Found {len(db_entries)} relevant examples in database")
                stats["used_database"] = True
                stats["database_samples"] = len(db_entries)

                # Train on database examples
                for entry in db_entries:
                    feed_sense_input(self.simulation.model, entry.data_type, entry.data)
                    for _ in range(steps):
                        self.simulation.step()

                print("Incorporated database knowledge into training")
            else:
                print("No relevant examples found in database")

        return stats


def populate_sample_knowledge(db_path: str = "knowledge.db") -> None:
    """Populate database with sample training data.

    Args:
        db_path: Path to database
    """
    print("Populating knowledge database with sample data...")

    db = KnowledgeDatabase(db_path)

    # Create sample vision patterns
    vision_entries = []
    for i in range(50):
        # Create simple patterns
        pattern = np.zeros((20, 20))
        if i % 4 == 0:
            pattern[::2, :] = 1.0  # Horizontal stripes
        elif i % 4 == 1:
            pattern[:, ::2] = 1.0  # Vertical stripes
        elif i % 4 == 2:
            np.fill_diagonal(pattern, 1.0)  # Diagonal
        else:
            pattern[::2, ::2] = 1.0  # Checkerboard
            pattern[1::2, 1::2] = 1.0

        # Add noise
        pattern += np.random.normal(0, 0.1, pattern.shape)
        pattern = np.clip(pattern, 0, 1)

        entry = KnowledgeEntry(
            id=None,
            category="pattern_recognition",
            data_type="vision",
            data=pattern,
            label=i % 4,
            metadata={"pattern_type": ["horizontal", "vertical", "diagonal", "checkerboard"][i % 4]},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        vision_entries.append(entry)

    db.add_batch(vision_entries)
    print(f"Added {len(vision_entries)} vision patterns")

    # Create sample digital sequences
    digital_entries = []
    for i in range(30):
        # Create simple sequences
        sequence = np.random.randint(0, 8, size=5)
        encoding = np.zeros((20, 20))
        encoding[:8, :5] = 0.0
        for j, val in enumerate(sequence):
            encoding[val, j] = 1.0

        entry = KnowledgeEntry(
            id=None,
            category="sequence_learning",
            data_type="digital",
            data=encoding,
            label=sequence.tolist(),
            metadata={"sequence_length": len(sequence)},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        digital_entries.append(entry)

    db.add_batch(digital_entries)
    print(f"Added {len(digital_entries)} digital sequences")

    # Print summary
    total = db.count()
    print(f"\nDatabase now contains {total} total entries:")
    print(f"  Pattern recognition: {db.count(category='pattern_recognition')}")
    print(f"  Sequence learning: {db.count(category='sequence_learning')}")

    db.close()
    print("Database populated successfully!")
