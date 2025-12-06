# 4D-Neural-Cognition

Dieses Modell implementiert ein 4D-Hirnsystem, das biologische Prinzipien mit digitalen Erweiterungen verbindet. Es simuliert Neuronen in einem vierdimensionalen Gitter, die altern, sterben und sich mit Vererbung mutierter Eigenschaften reproduzieren können. Verschiedene Hirnareale verarbeiten spezifische Sinne – inklusive eines digitalen Sinnes für Systemdaten und Muster.

## Features

- **4D Neuronengitter**: Neuronen in einem (x, y, z, w) Koordinatensystem
- **Leaky Integrate-and-Fire Modell**: Biophysikalisch inspirierte Neuronen mit Membranpotential
- **Zell-Lebenszyklus**: Alterung, Tod und Reproduktion mit Vererbung mutierter Eigenschaften
- **Hirnareale & Sinne**: Vision, Audition, Somatosensorik, Geschmack, Geruch, Vestibulär, Digital
- **Hebbsche Plastizität**: "Cells that fire together, wire together" Lernregel
- **Speicherung**: JSON für Konfiguration, HDF4 für effiziente Datenspeicherung

## Installation

```bash
pip install -r requirements.txt
```

Für HDF4-Unterstützung:
```bash
pip install pyhdf
```

## Verwendung

### Beispiel ausführen

```bash
python example.py
```

### Programmatische Nutzung

```python
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input, create_digital_sense_input
import numpy as np

# Modell laden
model = BrainModel(config_path='brain_base_model.json')

# Simulation initialisieren
sim = Simulation(model, seed=42)

# Neuronen in Arealen erstellen
sim.initialize_neurons(area_names=['V1_like', 'Digital_sensor'], density=0.1)

# Synaptische Verbindungen erstellen
sim.initialize_random_synapses(connection_probability=0.01)

# Sensorische Eingabe vorbereiten
vision_input = np.random.rand(20, 20) * 10
digital_input = create_digital_sense_input("Hello, World!")

# Simulation ausführen
for step in range(100):
    if step % 10 == 0:
        feed_sense_input(model, 'vision', vision_input)
        feed_sense_input(model, 'digital', digital_input)
    stats = sim.step()
    print(f"Step {step}: {len(stats['spikes'])} spikes")
```

## Projektstruktur

```
├── brain_base_model.json  # Konfiguration des Basismodells
├── example.py             # Beispielskript
├── requirements.txt       # Python-Abhängigkeiten
└── src/
    ├── __init__.py        # Package-Initialisierung
    ├── brain_model.py     # Neuron- und Synapse-Datenstrukturen
    ├── cell_lifecycle.py  # Zelltod und Vererbung
    ├── hdf4_storage.py    # HDF4/JSON Speicherung
    ├── plasticity.py      # Hebbsche Plastizitätsregeln
    ├── senses.py          # Sinneseingabe-Verarbeitung
    └── simulation.py      # Hauptsimulationsschleife
```

## Konfiguration

Die `brain_base_model.json` enthält:

- **lattice_shape**: Größe des 4D-Gitters [x, y, z, w]
- **neuron_model**: LIF-Parameter (tau_m, v_rest, v_reset, v_threshold)
- **cell_lifecycle**: Lebenszyklusparameter (max_age, health_decay, Mutationsraten)
- **plasticity**: Lernparameter (learning_rate, weight_bounds)
- **senses**: Sinneskonfiguration mit Arealen und Eingabegrößen
- **areas**: Koordinatenbereiche für jedes Hirnareal
