# 4D Neural Cognition - Projektübersicht

## Kurzbeschreibung

**4D Neural Cognition** ist ein experimentelles Forschungsframework zur Untersuchung neuromorpher Architekturen mit vier-dimensionaler räumlicher Organisation. Es kombiniert biologisch inspirierte Mechanismen (lokale Lernregeln, Zell-Lebenszyklen) mit neuartigen Organisationsprinzipien (kontinuierliche 4D-Topologie).

### Kernidee in einem Satz

*Kann eine kontinuierliche vier-dimensionale Gitterstruktur für neuronale Netze Vorteile gegenüber klassischen Schichtarchitekturen bieten?*

---

## 🎯 Forschungsziel

### Primäres Ziel

Wissenschaftliche Charakterisierung der **Continuous Spatial Intelligence** – eines Paradigmas, bei dem kognitive Strukturen durch räumliche Organisation in vier Dimensionen emergieren, ohne explizite Schicht-Definition.

### Sekundäre Ziele

1. **Vergleichende Evaluation**: Systematischer Vergleich mit etablierten Ansätzen (Deep Learning, SNNs, Reservoir Computing)
2. **Hypothesen-Validierung**: Testen von 10 formalisierten Hypothesen über 4D-Eigenschaften
3. **Werkzeug-Entwicklung**: Bereitstellung eines Frameworks für Forscher in Neuromorphik und AGI

---

## 🧬 Was macht diesen Ansatz einzigartig?

### 1. Vier-Dimensionale Organisation

**Traditionelle ANNs:** Diskrete Schichten (Input → Hidden → Output)  
**4D Neural Cognition:** Kontinuierliches 4D-Gitter (x, y, z, w)

```
Klassisch:           4D Approach:
┌─────┐              ┌─────────────────┐
│Input│              │  Continuous     │
├─────┤              │  4D Lattice     │
│Hidden│             │  (x,y,z,w)      │
├─────┤              │  - No layers    │
│Output│             │  - Spatial org. │
└─────┘              │  - w = hierarchy│
                     └─────────────────┘
```

**w-Dimension** als Meta-Koordinate:
- w=0: Sensorische Verarbeitung
- w=mittel: Intermediate Repräsentationen
- w=hoch: Abstrakte Konzepte

### 2. Biologisch Inspirierte Dynamik

| Feature | Biologisches Vorbild | Umsetzung im Modell | Zweck |
|---------|---------------------|---------------------|-------|
| **Lokale Plastizität** | Hebbian Learning, STDP | Lokale Gewichtsanpassung | Lernen ohne Backprop |
| **Zell-Lebenszyklus** | Neurogenese (abstrahiert) | Aging, Death, Reproduction | Langzeit-Adaption |
| **Neuromodulation** | Dopamin, Serotonin | Globale Modulatoren | Zustandsabhängiges Lernen |
| **Spiking Neurons** | Aktionspotentiale | LIF, Izhikevich | Zeitliche Präzision |

**Wichtig:** Dies sind **Inspirationen**, keine biologischen Simulationen (siehe `limitations.md`).

### 3. Räumliche Intelligenz

**Hypothese:** Räumliche Nähe im 4D-Gitter entspricht funktionaler Ähnlichkeit.

**Mechanismus:**
- Neuronen mit ähnlichen Funktionen clustern räumlich
- Emergenz von Arealen ohne explizite Vorgabe
- Ähnlich zu kortikalen Säulen, aber in 4D

**Testbar:** Siehe `docs/06-experiments/metrics.md`

---

## 🏗️ Architektur-Überblick

### Kern-Komponenten

```
┌──────────────────────────────────────────────────┐
│           Forschungs-Interface                    │
│  ┌────────────┬──────────────┬─────────────────┐ │
│  │Experiments │  Benchmarks  │  Analysis Tools │ │
│  └────────────┴──────────────┴─────────────────┘ │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│            Simulation Engine                      │
│  ┌──────────────────────────────────────────┐   │
│  │  4D Brain Model                          │   │
│  │  - Neurons (x,y,z,w coordinates)         │   │
│  │  - Synapses (weighted connections)       │   │
│  │  - Configuration (JSON-based)            │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Dynamics & Learning                     │   │
│  │  - Neuron Models (LIF, Izhikevich)       │   │
│  │  - Plasticity (Hebbian, STDP)            │   │
│  │  - Cell Lifecycle (Aging, Reproduction)  │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │  Sensory & Motor Systems                 │   │
│  │  - Vision, Audition, Touch, Digital      │   │
│  │  - Input mapping to 4D areas             │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Datenfluss

```
Sensor Input (Image, Sound, Text)
    │
    ▼
Preprocessing (Normalization, Reshaping)
    │
    ▼
Mapping to 4D Area (e.g., V1_like at w=0)
    │
    ▼
Neuron Activation (LIF Dynamics)
    │
    ▼
Spike Propagation (through Synapses)
    │
    ▼
Plasticity Update (Hebbian/STDP)
    │
    ▼
Cell Lifecycle (Aging, Reproduction with Mutation)
    │
    ▼
Output Readout (from Motor Area)
```

---

## 📊 Aktueller Stand

### Technische Reife

| Komponente | Status | Anmerkung |
|-----------|--------|-----------|
| **4D-Gitter** | ✅ Stabil | Getestet bis 100K Neuronen |
| **Neuronmodelle** | ✅ Stabil | LIF, Izhikevich validiert |
| **Plastizität** | ✅ Stabil | Hebbian, STDP implementiert |
| **Zell-Lebenszyklus** | 🟡 Experimentell | Funktional, nicht bio-validiert |
| **Benchmark-Framework** | 🟡 Experimentell | Erste Experimente |
| **GPU-Acceleration** | 🔴 In Entwicklung | Prototyp vorhanden |

Detailliert: `docs/00-meta/status.md`

### Wissenschaftliche Reife

- **Formalisierte Hypothesen**: 10 testbare Hypothesen
- **Publikationen**: 0 (in Vorbereitung)
- **Externe Validierung**: Ausstehend
- **Reproduzierbarkeit**: Config-System, Checkpoints vorhanden

---

## 🧪 Forschungsfragen

### Zentrale Fragen

1. **Räumliche Organisation**  
   Emergiert funktionale Spezialisierung in 4D-Gittern ohne explizite Layer-Definition?

2. **w-Dimension**  
   Kann die vierte Dimension als Organisationsprinzip für zeitliche/hierarchische Strukturen dienen?

3. **Lokales Lernen**  
   Führen lokale Plastizitätsregeln (ohne Backpropagation) zu vergleichbarer Leistung?

4. **Biologische Plausibilität**  
   Reproduziert das System bekannte neuronale Phänomene (Oszillationen, Travelling Waves)?

5. **Skalierung**  
   Wie skaliert Performance mit Netzwerkgröße im 4D-Raum?

Hypothesen: `docs/SCIENTIFIC_HYPOTHESES.md`

---

## 🎓 Anwendungsfälle

### Primär: Forschung

**Computational Neuroscience:**
- Test von Hypothesen über neuronale Organisation
- Emergenz kognitiver Strukturen
- Alternative zu Schicht-Architekturen

**AGI Research:**
- Exploration kontinuierlicher räumlicher Intelligenz
- Biologisch plausible Lernmechanismen
- Langzeit-Adaption durch Evolution

**Neuromorphic Computing:**
- Software-Prototyping für neuromorphe Hardware
- Benchmark-Entwicklung für 4D-Architekturen

### Sekundär: Anwendungen

**Proof-of-Concept:**
- Spatio-temporale Muster-Erkennung
- Multi-sensorische Integration
- Zeitreihen-Vorhersage mit w-Dimension

**Nicht:** Produktions-Deployments (siehe `research-scope.md`)

---

## 📚 Dokumentations-Struktur

### Navigations-Hilfe

```
docs/
│
├── 00-meta/              ← Projektsteuerung
│   ├── vision.md         ← START HIER
│   ├── research-scope.md ← Was wird/wird nicht untersucht
│   ├── roadmap.md
│   └── status.md         ← Komponenten-Reife
│
├── 01-overview/          ← DU BIST HIER
│   ├── index.md          ← Diese Datei
│   ├── glossary.md       ← Begriffsdefinitionen
│   ├── prior-art.md      ← Abgrenzung zu anderen Ansätzen
│   └── assumptions.md    ← Grundannahmen
│
├── 02-theoretical-foundation/  ← Theorie
├── 03-neural-architecture/     ← Formales Modell
├── 04-dynamics-and-learning/   ← Lernen & Emergenz
├── 05-implementation/          ← Technische Umsetzung
├── 06-experiments/             ← Benchmarks & Evaluation
├── 07-decisions/               ← Architecture Decision Records
└── 99-appendix/                ← Referenzen, Vergleiche
```

---

## 🚀 Quick Start für Forscher

### 1. Verständnis aufbauen

Empfohlene Lese-Reihenfolge:
1. Diese Datei (`index.md`)
2. `glossary.md` - Begriffe klären
3. `assumptions.md` - Grundannahmen verstehen
4. `docs/02-theoretical-foundation/` - Theoretischer Hintergrund
5. `docs/03-neural-architecture/` - Formales Modell
6. `docs/06-experiments/` - Wie man testet

### 2. Installation & Erste Schritte

```bash
# Clone Repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

# Virtual Environment
python -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Beispiel ausführen
python example.py

# Web-Interface starten
python app.py
```

Detailliert: `docs/user-guide/INSTALLATION.md`

### 3. Eigene Experimente

```python
from src.brain_model import BrainModel
from src.simulation import Simulation

# Modell erstellen
model = BrainModel("configs/small_4d_brain.json")

# Simulation initialisieren
sim = Simulation(model, seed=42)

# Training
for step in range(1000):
    metrics = sim.step()
    if step % 100 == 0:
        print(f"Step {step}: {metrics}")

# Speichern
model.save("my_experiment.h5")
```

Tutorials: `docs/tutorials/`

---

## 🤝 Mitarbeit

### Für Forscher

- **Hypothesen testen**: Framework nutzen für eigene Experimente
- **Benchmarks erweitern**: Neue Tasks beitragen
- **Vergleichsstudien**: Vergleich mit eigenen Modellen

Siehe: `docs/00-meta/contribution-model.md`

### Für Studenten

- **Abschlussarbeiten**: Forschungsfragen verfügbar
- **Code-Beiträge**: Issues auf GitHub
- **Dokumentation**: Verbesserungsvorschläge willkommen

### Kontakt

- **Maintainer**: Thomas Heisig
- **E-Mail**: t_heisig@gmx.de
- **Location**: Ganderkesee, Germany
- **GitHub**: [Issues](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues)

---

## 📖 Wichtige Dokumente

### Muss lesen (für Forscher)

- `docs/00-meta/research-scope.md` - Was wird/wird nicht untersucht
- `docs/01-overview/assumptions.md` - Grundannahmen des Modells
- `docs/02-theoretical-foundation/limitations.md` - Bewusste Vereinfachungen
- `docs/06-experiments/metrics.md` - Wie Erfolg gemessen wird
- `docs/99-appendix/open-questions.md` - Ungelöste Probleme

### Für Entwickler

- `CONTRIBUTING.md` - Contribution Guidelines
- `docs/developer-guide/coding-standards.md` - Code-Standards
- `docs/05-implementation/system-overview.md` - Architektur-Details

### Für Anwender

- `README.md` - Projekt-Übersicht
- `docs/user-guide/INSTALLATION.md` - Setup
- `docs/tutorials/GETTING_STARTED.md` - Erste Schritte

---

## ⚠️ Wichtige Hinweise

### Was dieses Projekt NICHT ist

- ❌ **Kein fertiges Produkt**: Forschungs-Prototyp
- ❌ **Keine biologische Simulation**: Abstrahiertes Modell
- ❌ **Kein AGI-System**: Exploration eines Ansatzes
- ❌ **Keine Überlegenheitsbehauptung**: Charakterisierung, nicht Marketing

### Was dieses Projekt IST

- ✅ **Forschungswerkzeug**: Für Hypothesen-Tests
- ✅ **Open Source**: MIT Lizenz, vollständig transparent
- ✅ **Wissenschaftlich**: Reproduzierbar, dokumentiert
- ✅ **Explorativ**: Offene Fragen, negative Ergebnisse willkommen

---

## 📜 Lizenz & Nutzung

**Lizenz:** MIT (siehe `LICENSE`)

**Verwendung:**
- ✅ Akademische Forschung
- ✅ Studenten-Projekte
- ✅ Open-Source-Entwicklung
- ⚠️ Kommerzielle Nutzung: möglich, aber keine Garantie

**Citation:**
```bibtex
@software{4d_neural_cognition,
  title = {4D Neural Cognition: A Neuromorphic AI Framework},
  author = {Heisig, Thomas},
  year = {2025},
  url = {https://github.com/Thomas-Heisig/4D-Neural-Cognition}
}
```

---

## 🔗 Externe Ressourcen

### Verwandte Projekte

- [NEST Simulator](https://www.nest-simulator.org/) - Spiking Neural Networks
- [Brian2](https://brian2.readthedocs.io/) - Neuron Simulator
- [ANNarchy](https://annarchy.github.io/) - Artificial Neural Networks
- [Nengo](https://www.nengo.ai/) - Neuromorphic Computing

### Konferenzen

- COSYNE (Computational and Systems Neuroscience)
- CNS (Computational Neuroscience Meeting)
- NeurIPS, ICML (Machine Learning)
- ICONS (Neuromorphic Systems)

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Für Fragen: t_heisig@gmx.de*
