# 4D-Neural-Cognition

Dieses Modell implementiert ein 4D-Hirnsystem, das biologische Prinzipien mit digitalen Erweiterungen verbindet. Es simuliert Neuronen in einem vierdimensionalen Gitter, die altern, sterben und sich mit Vererbung mutierter Eigenschaften reproduzieren können. Verschiedene Hirnareale verarbeiten spezifische Sinne – inklusive eines digitalen Sinnes für Systemdaten und Muster.

## Features

- **4D Neuronengitter**: Neuronen in einem (x, y, z, w) Koordinatensystem
- **Leaky Integrate-and-Fire Modell**: Biophysikalisch inspirierte Neuronen mit Membranpotential
- **Zell-Lebenszyklus**: Alterung, Tod und Reproduktion mit Vererbung mutierter Eigenschaften
- **Hirnareale & Sinne**: Vision, Audition, Somatosensorik, Geschmack, Geruch, Vestibulär, Digital
- **Hebbsche Plastizität**: "Cells that fire together, wire together" Lernregel
- **Speicherung**: JSON für Konfiguration, HDF5 für effiziente Datenspeicherung (mit Kompression)
- **Web-Frontend**: Modernes Browser-Interface mit Echtzeit-Visualisierung und Logging

## Installation

```bash
pip install -r requirements.txt
```



## Verwendung

### Web-Frontend (empfohlen)

Starten Sie die Web-Anwendung für eine benutzerfreundliche grafische Oberfläche:

```bash
python app.py
```

Öffnen Sie dann einen Browser und navigieren Sie zu `http://localhost:5000`.

Das Frontend bietet:
- 🎮 **Modell-Steuerung**: Initialisierung und Konfiguration
- 🔥 **Heatmap-Visualisierung**: Echtzeit-Darstellung von Input-, Hidden- und Output-Layern
- 💻 **Terminal**: Input/Output für sensorische Daten
- 💬 **Chat-Interface**: Interaktive Befehle und Operationen
- 📋 **Logging**: Vollständige Protokollierung aller Systemereignisse
- ⚡ **Training**: Start/Stop-Kontrolle für Simulationsläufe

### Kommandozeilen-Beispiel

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
├── example.py             # Kommandozeilen-Beispielskript
├── app.py                 # Flask Web-Anwendung
├── requirements.txt       # Python-Abhängigkeiten
├── templates/
│   └── index.html        # Web-Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css     # Modernes UI-Styling
│   └── js/
│       └── app.js        # Frontend JavaScript
└── src/
    ├── __init__.py        # Package-Initialisierung
    ├── brain_model.py     # Neuron- und Synapse-Datenstrukturen
    ├── cell_lifecycle.py  # Zelltod und Vererbung
    ├── storage.py         # HDF5/JSON Speicherung
    ├── plasticity.py      # Hebbsche Plastizitätsregeln
    ├── senses.py          # Sinneseingabe-Verarbeitung
    └── simulation.py      # Hauptsimulationsschleife
```

## Web-Frontend Features

Das moderne Web-Interface bietet folgende Funktionen:

### 🎮 Modell-Steuerung
- Initialisierung neuer Modelle
- Konfiguration von Neuronen und Synapsen
- Einstellung der Neuronendichte
- Anzeige von Modell-Informationen

### 🔥 Heatmap-Visualisierung
- Echtzeit-Darstellung der neuronalen Aktivität
- Separate Ansichten für Input-, Hidden- und Output-Layer
- Farbcodierte Membranpotential-Darstellung

### 💻 Input/Output Terminal
- Eingabe sensorischer Daten (Vision, Audition, Digital, etc.)
- Text-basierte Eingabe für Digital-Sense
- Array-Eingabe für andere Sinnesmodalitäten
- Echtzeit-Feedback zu Operationen

### 💬 Chat & Operationen
- Interaktive Befehle für Systemsteuerung
- Verfügbare Befehle: `help`, `info`, `status`, `init`, `step`, `run`
- Sofortige Rückmeldung zu allen Operationen

### 📋 System Logging
- Vollständige Protokollierung aller Ereignisse
- Filterung nach Log-Level (INFO, WARNING, ERROR, SUCCESS)
- WebSocket-basierte Echtzeit-Updates
- Exportierbar für Analyse

### ⚡ Training & Simulation
- Einzelschritte oder Multi-Step-Training
- Start/Stop-Kontrolle während des Trainings
- Fortschrittsanzeige mit Live-Updates
- Automatische Heatmap-Aktualisierung

### 💾 Speichern & Laden
- Export als JSON (lesbar) oder HDF5 (komprimiert)
- Laden bestehender Modelle
- Zustandserhaltung zwischen Sessions

## Konfiguration

Die `brain_base_model.json` enthält:

- **lattice_shape**: Größe des 4D-Gitters [x, y, z, w]
- **neuron_model**: LIF-Parameter (tau_m, v_rest, v_reset, v_threshold)
- **cell_lifecycle**: Lebenszyklusparameter (max_age, health_decay, Mutationsraten)
- **plasticity**: Lernparameter (learning_rate, weight_bounds)
- **senses**: Sinneskonfiguration mit Arealen und Eingabegrößen
- **areas**: Koordinatenbereiche für jedes Hirnareal

## Technologie-Stack

- **Backend**: Flask (Python Web-Framework)
- **Frontend**: Vanilla JavaScript mit Socket.IO
- **Styling**: Modernes CSS mit Dark Theme
- **Visualisierung**: HTML5 Canvas für Heatmaps
- **Datenspeicherung**: HDF5 (statt veraltetem HDF4)
- **Echtzeit-Kommunikation**: WebSocket (Flask-SocketIO)
