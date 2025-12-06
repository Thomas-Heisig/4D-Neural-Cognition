# Knowledge Database Guide

The Knowledge Database system allows you to store training data and knowledge that the neural network can access and learn from, even before it has learned from direct experience.

## Table of Contents

1. [Overview](#overview)
2. [Basic Usage](#basic-usage)
3. [Creating Entries](#creating-entries)
4. [Querying Data](#querying-data)
5. [Training with Knowledge](#training-with-knowledge)
6. [Advanced Features](#advanced-features)
7. [Examples](#examples)

---

## Overview

### What is the Knowledge Database?

The Knowledge Database is a SQLite-based storage system for:
- **Training patterns**: Pre-defined patterns for network training
- **Reference data**: Known inputs and their expected outputs
- **Experience replay**: Storing past experiences for later learning
- **Transfer learning**: Sharing knowledge across simulations

### Key Benefits

- **Persistent storage**: Data survives across sessions
- **Fast queries**: Indexed for quick retrieval
- **Flexible metadata**: Store arbitrary additional information
- **Batch operations**: Efficiently store and retrieve multiple entries

---

## Basic Usage

### Importing and Creating a Database

```python
from knowledge_db import KnowledgeDatabase, KnowledgeEntry
import numpy as np
from datetime import datetime

# Create or connect to database
db = KnowledgeDatabase(db_path="my_knowledge.db")
```

### Adding a Simple Entry

```python
# Create a knowledge entry
entry = KnowledgeEntry(
    id=None,  # Auto-assigned by database
    category="pattern"
    data_type="vision"
    data=np.eye(10),  # A diagonal pattern
    label="diagonal"
    metadata={"difficulty": "easy", "size": 10}
    timestamp=datetime.now().isoformat()
)

# Add to database
entry_id = db.add_entry(entry)
print(f"Stored entry with ID: {entry_id}")
```

### Retrieving an Entry

```python
# Get entry by ID
retrieved = db.get_entry(entry_id)

print(f"Category: {retrieved.category}")
print(f"Data type: {retrieved.data_type}")
print(f"Label: {retrieved.label}")
print(f"Data shape: {retrieved.data.shape}")
```

---

## Creating Entries

### Vision Patterns

```python
# Store various visual patterns
patterns = {
    "horizontal_line": np.zeros((10, 10))
    "vertical_line": np.zeros((10, 10))
    "diagonal": np.eye(10)
    "checkerboard": None  # Will create below
}

patterns["horizontal_line"][5, :] = 1.0
patterns["vertical_line"][:, 5] = 1.0

# Checkerboard
x, y = np.meshgrid(range(10), range(10))
patterns["checkerboard"] = ((x + y) % 2).astype(float)

# Store all patterns
pattern_ids = {}
for name, pattern in patterns.items():
    entry = KnowledgeEntry(
        id=None
        category="visual_pattern"
        data_type="vision"
        data=pattern
        label=name
        metadata={"created": datetime.now().isoformat()}
        timestamp=datetime.now().isoformat()
    )
    pattern_ids[name] = db.add_entry(entry)

print(f"Stored {len(pattern_ids)} patterns")
```

### Audio Sequences

```python
# Store audio patterns (frequency over time)
# Example: Rising tone
audio = np.zeros((20, 10))
for t in range(10):
    audio[t*2, t] = 1.0

entry = KnowledgeEntry(
    id=None
    category="audio_pattern"
    data_type="audio"
    data=audio
    label="rising_tone"
    metadata={"frequency_bins": 20, "duration": 10}
    timestamp=datetime.now().isoformat()
)

db.add_entry(entry)
```

### Digital/Text Data

```python
from senses import create_digital_sense_input

# Store text patterns
texts = ["hello", "world", "neural", "network"]

for text in texts:
    digital_data = create_digital_sense_input(text)
    
    entry = KnowledgeEntry(
        id=None
        category="text_pattern"
        data_type="digital"
        data=digital_data
        label=text
        metadata={"length": len(text)}
        timestamp=datetime.now().isoformat()
    )
    
    db.add_entry(entry)
```

### Labeled Training Data

```python
# Store input-output pairs for supervised learning
training_pairs = [
    (np.eye(5), "class_A")
    (np.ones((5, 5)), "class_B")
    (np.random.rand(5, 5), "class_C")
]

for data, label in training_pairs:
    entry = KnowledgeEntry(
        id=None
        category="training_data"
        data_type="vision"
        data=data
        label=label
        metadata={"training_phase": "initial"}
        timestamp=datetime.now().isoformat()
    )
    
    db.add_entry(entry)
```

---

## Querying Data

### Get All Entries

```python
all_entries = db.get_all_entries()
print(f"Total entries: {len(all_entries)}")

for entry in all_entries[:5]:  # Show first 5
    print(f"ID: {entry.id}, Category: {entry.category}, Label: {entry.label}")
```

### Query by Category

```python
# Get all visual patterns
visual_patterns = db.query_entries(category="visual_pattern")

print(f"Found {len(visual_patterns)} visual patterns")
for pattern in visual_patterns:
    print(f"  - {pattern.label}")
```

### Query by Data Type

```python
# Get all vision data
vision_data = db.query_entries(data_type="vision")
print(f"Vision entries: {len(vision_data)}")

# Get all audio data
audio_data = db.query_entries(data_type="audio")
print(f"Audio entries: {len(audio_data)}")
```

### Query by Both

```python
# Get visual training data specifically
visual_training = db.query_entries(
    category="training_data"
    data_type="vision"
)
print(f"Visual training samples: {len(visual_training)}")
```

### Query by Label

```python
# Find specific labeled entries
diagonal_entries = db.query_by_label("diagonal")
print(f"Diagonal patterns: {len(diagonal_entries)}")
```

### Random Sampling

```python
# Get random entries for batch training
batch = db.get_random_batch(batch_size=10, category="training_data")

print(f"Sampled {len(batch)} entries for training")
for entry in batch:
    print(f"  Label: {entry.label}, Shape: {entry.data.shape}")
```

---

## Training with Knowledge

### Basic Training Loop

```python
from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input

# Setup
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
sim.initialize_neurons(areas=["V1_like"], density=0.1)
sim.initialize_random_synapses(connection_prob=0.1)

# Create database
db = KnowledgeDatabase("training_data.db")

# Get training data
training_data = db.query_entries(category="training_data")

# Train on knowledge
print("Training from knowledge database...")
for epoch in range(5):
    print(f"Epoch {epoch+1}")
    
    for entry in training_data:
        # Feed data to network
        feed_sense_input(
            sim.model
            sense_name=entry.data_type
            input_data=entry.data
            
        )
        
        # Process
        sim.run(steps=30)
        
        # Learn
        sim.apply_plasticity()

print("Training complete!")
```

### Batch Training

```python
def train_batch(sim, db, batch_size=10, num_batches=50):
    """Train on random batches from database."""
    
    for batch_num in range(num_batches):
        # Get random batch
        batch = db.get_random_batch(
            batch_size=batch_size
            category="training_data"
        )
        
        # Train on batch
        for entry in batch:
            feed_sense_input(
                sim.model
                sense_name=entry.data_type
                input_data=entry.data
                
            )
            sim.run(steps=20)
            sim.apply_plasticity()
        
        if batch_num % 10 == 0:
            print(f"Completed batch {batch_num}/{num_batches}")

train_batch(sim, db)
```

### Curriculum Learning with Database

```python
def curriculum_training(sim, db):
    """Train from simple to complex using metadata."""
    
    # Get all training data
    all_data = db.query_entries(category="training_data")
    
    # Sort by difficulty (stored in metadata)
    sorted_data = sorted(
        all_data
        key=lambda x: x.metadata.get("difficulty", 0)
    )
    
    # Train in order
    for entry in sorted_data:
        print(f"Training on: {entry.label} (difficulty: {entry.metadata.get('difficulty')})")
        
        feed_sense_input(
            sim.model
            sense_name=entry.data_type
            input_data=entry.data
            
        )
        
        sim.run(steps=30)
        sim.apply_plasticity()

curriculum_training(sim, db)
```

### Experience Replay

```python
def store_experience(db, sensory_input, outcome, category="experience"):
    """Store an experience for later replay."""
    
    entry = KnowledgeEntry(
        id=None
        category=category
        data_type="vision"
        data=sensory_input
        label=outcome
        metadata={
            "reward": outcome
            "stored_at": datetime.now().isoformat()
        }
        timestamp=datetime.now().isoformat()
    )
    
    return db.add_entry(entry)

def replay_experiences(sim, db, num_replays=20):
    """Replay past experiences for additional learning."""
    
    # Get experiences
    experiences = db.query_entries(category="experience")
    
    # Replay random sample
    import random
    replay_sample = random.sample(experiences, min(num_replays, len(experiences)))
    
    for exp in replay_sample:
        feed_sense_input(
            sim.model
            sense_name=exp.data_type
            input_data=exp.data
            
        )
        sim.run(steps=20)
        sim.apply_plasticity()

# During simulation
outcome = 1.0  # Positive outcome
store_experience(db, current_input, outcome)

# Later
replay_experiences(sim, db)
```

---

## Advanced Features

### Database Statistics

```python
stats = db.get_statistics()

print(f"Total entries: {stats['total_entries']}")
print(f"Categories: {stats['categories']}")
print(f"Data types: {stats['data_types']}")
print(f"Entries by category:")
for cat, count in stats['entries_by_category'].items():
    print(f"  {cat}: {count}")
```

### Bulk Operations

```python
# Add multiple entries at once
entries = []
for i in range(100):
    data = np.random.rand(10, 10)
    entry = KnowledgeEntry(
        id=None
        category="random_data"
        data_type="vision"
        data=data
        label=f"random_{i}"
        metadata={"index": i}
        timestamp=datetime.now().isoformat()
    )
    entries.append(entry)

# Bulk add (if implemented)
# db.add_entries_bulk(entries)

# Or add in loop
for entry in entries:
    db.add_entry(entry)
```

### Exporting Data

```python
# Export entries to files
import json

entries = db.query_entries(category="visual_pattern")

for entry in entries:
    # Save data as numpy
    np.save(f"pattern_{entry.label}.npy", entry.data)
    
    # Save metadata as JSON
    with open(f"pattern_{entry.label}_meta.json", 'w') as f:
        json.dump({
            'category': entry.category
            'data_type': entry.data_type
            'label': entry.label
            'metadata': entry.metadata
            'timestamp': entry.timestamp
        }, f, indent=2)
```

### Deleting Entries

```python
# Delete by ID
db.delete_entry(entry_id)

# Delete by category
entries_to_delete = db.query_entries(category="temporary")
for entry in entries_to_delete:
    db.delete_entry(entry.id)
```

### Updating Entries

```python
# Get entry
entry = db.get_entry(entry_id)

# Modify
entry.metadata["updated"] = datetime.now().isoformat()
entry.label = "updated_label"

# Update in database
db.update_entry(entry)
```

---

## Examples

### Example 1: Building a Training Set

```python
#!/usr/bin/env python3
"""Build a training set in the knowledge database."""

import numpy as np
from knowledge_db import KnowledgeDatabase, KnowledgeEntry
from datetime import datetime

def create_training_set():
    db = KnowledgeDatabase("training_set.db")
    
    # Create 50 varied patterns
    print("Creating training set...")
    
    for i in range(50):
        # Random pattern
        pattern = np.random.rand(10, 10)
        
        # Add structure
        if i % 3 == 0:
            pattern[5, :] = 1.0  # Horizontal line
            label = "horizontal"
        elif i % 3 == 1:
            pattern[:, 5] = 1.0  # Vertical line
            label = "vertical"
        else:
            pattern = np.eye(10)  # Diagonal
            label = "diagonal"
        
        entry = KnowledgeEntry(
            id=None
            category="training_pattern"
            data_type="vision"
            data=pattern
            label=label
            metadata={
                "index": i
                "difficulty": "medium"
                "created": datetime.now().isoformat()
            }
            timestamp=datetime.now().isoformat()
        )
        
        db.add_entry(entry)
    
    # Check statistics
    stats = db.get_statistics()
    print(f"Created {stats['total_entries']} entries")
    print(f"Categories: {list(stats['entries_by_category'].keys())}")
    
    db.close()

if __name__ == "__main__":
    create_training_set()
```

### Example 2: Using Database for Pre-training

```python
#!/usr/bin/env python3
"""Pre-train a network using knowledge database."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input
from knowledge_db import KnowledgeDatabase

def pretrain_network():
    # Setup network
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(areas=["V1_like"], density=0.15)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    # Load training data
    db = KnowledgeDatabase("training_set.db")
    training_data = db.query_entries(category="training_pattern")
    
    print(f"Pre-training on {len(training_data)} examples...")
    
    # Pre-training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        
        for i, entry in enumerate(training_data):
            # Feed pattern
            feed_sense_input(
                sim.model
                sense_name="vision"
                input_data=entry.data
                
            )
            
            # Process
            sim.run(steps=25)
            
            # Learn
            sim.apply_plasticity()
            
            if i % 10 == 0:
                print(f"  Processed {i}/{len(training_data)} patterns")
    
    # Save pre-trained model
    from storage import save_to_hdf5
    save_to_hdf5(model, "pretrained_model.h5")
    print("\nPre-trained model saved!")
    
    db.close()

if __name__ == "__main__":
    pretrain_network()
```

---

## Best Practices

1. **Organize by Category**: Use meaningful category names
2. **Add Metadata**: Store useful information for later filtering
3. **Label Clearly**: Use descriptive labels
4. **Regular Backups**: Back up your database files
5. **Close Connections**: Always close database when done
6. **Index Wisely**: Queries on category and data_type are optimized

---

## API Reference

See the full API documentation in the source code (`src/knowledge_db.py`) for:
- `KnowledgeDatabase` class methods
- `KnowledgeEntry` dataclass
- Query methods and filters
- Batch operations

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
