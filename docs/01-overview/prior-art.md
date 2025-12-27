# Prior Art - Abgrenzung zu bestehenden Ansätzen

## Zweck

Dieses Dokument positioniert 4D Neural Cognition im Kontext verwandter Forschungsfelder und erklärt, wie es sich von etablierten Ansätzen unterscheidet – **ohne Überlegenheitsbehauptungen**.

---

## 🔬 Verwandte Forschungsfelder

### 1. Deep Learning (PyTorch, TensorFlow, Keras)

**Gemeinsamkeiten:**
- Neuronale Netze als Grundstruktur
- Trainierbare Gewichte
- Mehrschichtige Verarbeitung

**Unterschiede:**

| Aspekt | Deep Learning | 4D Neural Cognition |
|--------|--------------|---------------------|
| **Architektur** | Diskrete Schichten (Layer) | Kontinuierliches 4D-Gitter |
| **Lernregel** | Backpropagation (global) | Lokale Plastizität (Hebbian, STDP) |
| **Topologie** | Vollvernetzt oder filter-basiert | Räumlich organisiert |
| **Zeitliche Dynamik** | Feedforward oder RNN | Spiking Neurons mit intrinsischer Dynamik |
| **Adaption** | Training-Phase getrennt | Kontinuierliche Adaption (Lifecycle) |

**Nicht behauptet:** 4D ist "besser" – es ist **anders** mit spezifischen Trade-offs.

**Vergleichs-Referenz:** `docs/99-appendix/comparisons.md`

---

### 2. Spiking Neural Networks (NEST, Brian2, ANNarchy)

**Gemeinsamkeiten:**
- Spiking Neurons (LIF, Izhikevich)
- Biologisch inspirierte Dynamik
- STDP und Hebbian Learning
- Zeitliche Präzision

**Unterschiede:**

| Aspekt | NEST/Brian2 | 4D Neural Cognition |
|--------|-------------|---------------------|
| **Fokus** | Biologische Genauigkeit | Räumliche Organisation in 4D |
| **Topologie** | Frei definierbar (oft biologisch) | Strukturiertes 4D-Gitter |
| **Zell-Lebenszyklus** | Statisch (oder explizit modelliert) | Intrinsisches Feature (Aging, Reproduction) |
| **w-Dimension** | Nicht vorhanden | Zentrale Organisationsdimension |
| **Einsatzgebiet** | Neurobiologie-Simulation | Hypothesen über 4D-Organisation |

**Brian2 vs. 4D:**
- Brian2: Flexibles Framework für beliebige Neuronmodelle
- 4D: Spezialisiert auf 4D-Gitter-Experimente

**NEST vs. 4D:**
- NEST: Optimiert für große biologisch realistische Netze
- 4D: Exploriert neuartige 4D-Topologien

---

### 3. Neuromorphic Hardware (Loihi, SpiNNaker, TrueNorth)

**Gemeinsamkeiten:**
- Spiking Neurons
- Energieeffizienz-Fokus
- Event-driven Processing

**Unterschiede:**

| Aspekt | Neuromorphic HW | 4D Neural Cognition |
|--------|-----------------|---------------------|
| **Medium** | Spezial-Hardware | Software-Simulation |
| **Ziel** | Deployment & Effizienz | Architektur-Exploration |
| **Flexibilität** | Hardware-constraints | Beliebige Modelle testbar |
| **4D-Struktur** | Nicht native | Kernkonzept |

**Komplementär:** 4D könnte als Software-Prototyp für neuromorphe Hardware dienen (siehe `docs/NEUROMORPHIC_HARDWARE_STRATEGY.md`).

---

### 4. Reservoir Computing (Echo State Networks, Liquid State Machines)

**Gemeinsamkeiten:**
- Fixed random connections
- Kein Training der Reservoir-Gewichte
- Emergente Dynamik

**Unterschiede:**

| Aspekt | Reservoir Computing | 4D Neural Cognition |
|--------|-------------------|---------------------|
| **Connections** | Random, statisch | Strukturiert (4D-basiert), plastisch |
| **Lernort** | Nur Readout trainiert | Plastizität im gesamten Netz |
| **Topologie** | Beliebig | 4D-Gitter |
| **Evolution** | Keine | Zell-Lebenszyklus mit Mutation |

**ESN vs. 4D:**
- ESN: Feste Reservoir-Dynamik
- 4D: Plastizität und Evolution im gesamten Netzwerk

---

### 5. Graph Neural Networks (GNNs)

**Gemeinsamkeiten:**
- Keine strikte Layer-Struktur
- Nachbarschafts-basierte Verarbeitung
- Flexible Topologie

**Unterschiede:**

| Aspekt | GNNs | 4D Neural Cognition |
|--------|------|---------------------|
| **Raum** | Abstrakter Graph | Euklidischer 4D-Raum |
| **Lernregel** | Backpropagation | Lokale Plastizität |
| **Biologische Inspiration** | Gering | Hoch (Spiking, Plasticity) |
| **Struktur** | Beliebiger Graph | Regelmäßiges 4D-Gitter |

---

### 6. HyperNEAT & NEAT

**Gemeinsamkeiten:**
- Topologie-Evolution
- Räumliche Koordinaten
- Emergence of structure

**Unterschiede:**

| Aspekt | HyperNEAT | 4D Neural Cognition |
|--------|-----------|---------------------|
| **Evolution** | Genetischer Algorithmus | Zell-Reproduktion mit Mutation |
| **Dynamik** | Statische Netze | Spiking Dynamics |
| **Lernmechanismus** | Evolution allein | Evolution + Plastizität |
| **Dimensionalität** | Beliebig | 4D-Fokus |

---

### 7. Neural Cellular Automata

**Gemeinsamkeiten:**
- Lokale Regeln
- Emergente Strukturen
- Räumliche Organisation

**Unterschiede:**

| Aspekt | Neural CA | 4D Neural Cognition |
|--------|-----------|---------------------|
| **Update-Regel** | Zelluläre Automaten-Regel | Neuronale Dynamik (LIF, Izhikevich) |
| **Lernbarkeit** | Regel trainierbar | Synaptic Plasticity |
| **Biologische Inspiration** | Morphogenesis | Neuroscience |
| **Anwendung** | Pattern Generation | Kognitive Verarbeitung |

---

### 8. Cortical Column Models (Blue Brain, Human Brain Project)

**Gemeinsamkeiten:**
- Biologische Detailtreue
- Räumliche Organisation
- Mehrschichtige Struktur

**Unterschiede:**

| Aspekt | Cortical Column | 4D Neural Cognition |
|--------|-----------------|---------------------|
| **Biologische Genauigkeit** | Sehr hoch (Morphologie, Biochemie) | Abstrahiert |
| **Skalierung** | Begrenzt (Rechenaufwand) | Skalierbar (Vereinfachungen) |
| **w-Dimension** | Nicht vorhanden (3D real) | Abstrakte 4. Dimension |
| **Ziel** | Gehirn-Simulation | Architektur-Hypothesen |

**Blue Brain Project:**
- Fokus: Biologische Korrektheit
- 4D: Fokus auf neuartige Organisationsprinzipien

---

## 🎯 Nischenpositionierung

### Wo 4D Neural Cognition einzigartig ist

**Kombination von:**
1. **4D-Gitter-Topologie** (nicht in anderen Frameworks)
2. **Lokale Plastizität + Zell-Evolution** (einzigartige Kombination)
3. **w-Dimension als Organisationsprinzip** (neuartig)
4. **Biologische Inspiration ohne bio-Simulation** (Balance)

### Wo 4D Neural Cognition **nicht** konkurriert

- ❌ **Nicht:** State-of-the-art auf Standard-ML-Benchmarks (ImageNet, etc.)
- ❌ **Nicht:** Präzise biologische Gehirn-Simulation
- ❌ **Nicht:** Produktions-ready Deep-Learning-Framework
- ❌ **Nicht:** Neuromorphic-Hardware-Ersatz

### Was 4D Neural Cognition **bietet**

- ✅ **Exploration**: Testbett für 4D-Organisationshypothesen
- ✅ **Forschungswerkzeug**: Framework für neuartige Experimente
- ✅ **Vergleichsplattform**: Systematischer Vergleich mit etablierten Methoden

---

## 📊 Vergleichstabelle (Zusammenfassung)

| Eigenschaft | Deep Learning | SNN (NEST) | Neuromorphic HW | Reservoir | 4D Neural Cog. |
|-------------|---------------|------------|-----------------|-----------|----------------|
| **Topologie** | Layers | Beliebig | HW-constrained | Random | **4D-Gitter** |
| **Lernregel** | Backprop | STDP | STDP | Readout | **Lokal** |
| **Zeitdynamik** | Diskret/RNN | Spiking | Spiking | Spiking | **Spiking** |
| **Evolution** | Nein | Nein | Nein | Nein | **Ja** |
| **w-Dimension** | Nein | Nein | Nein | Nein | **Ja** |
| **Bio-Plausibilität** | Niedrig | Hoch | Mittel | Mittel | **Mittel-Hoch** |
| **Skalierung** | Sehr gut | Gut | Gut | Gut | **Mittel** |
| **Flexibilität** | Hoch | Sehr hoch | Niedrig | Mittel | **Hoch** |

---

## 🔗 Potentielle Synergien

### Mit Deep Learning

**Hybride Ansätze:**
- 4D als Feature Extractor, DL als Classifier
- Transfer Learning zwischen 4D und DL
- Siehe: `docs/05-implementation/` - Framework Bridges

### Mit SNNs

**Ergänzung:**
- 4D-Topologie als Alternative zu bio-inspirierten Layouts
- Vergleichsstudien mit NEST/Brian2 auf gleichen Tasks

### Mit Neuromorphic Hardware

**Software-zu-Hardware-Pipeline:**
- 4D als Prototyping-Tool
- Deployment auf Loihi/SpiNNaker
- Siehe: `docs/NEUROMORPHIC_HARDWARE_STRATEGY.md`

---

## 📚 Relevante Literatur

### Vergleichsstudien

1. **SNNs vs. ANNs:**  
   Tavanaei et al. (2019). "Deep learning in spiking neural networks." Neural Networks.

2. **Reservoir Computing:**  
   Lukoševičius & Jaeger (2009). "Reservoir computing approaches to recurrent neural network training." Computer Science Review.

3. **Neuromorphic Computing:**  
   Davies et al. (2018). "Loihi: A neuromorphic manycore processor." IEEE Micro.

4. **Cortical Organization:**  
   Markram et al. (2015). "Reconstruction and Simulation of Neocortical Microcircuitry." Cell.

### Siehe auch

`docs/99-appendix/references.md` - Vollständige Literaturliste

---

## ⚖️ Ehrliche Einordnung

### Stärken (potentiell)

- ✅ Exploration neuartiger 4D-Organisation
- ✅ Kombinierter Ansatz (Plasticity + Evolution)
- ✅ Biologische Inspiration ohne Komplexität

### Schwächen (bekannt)

- ❌ Keine State-of-the-art-Performance auf Standard-Tasks
- ❌ Skalierung begrenzt (vs. Deep Learning Frameworks)
- ❌ Noch nicht extern validiert

### Offene Fragen

- ❓ Wann ist 4D-Organisation vorteilhaft?
- ❓ Optimal
es Verhältnis biologischer Realismus vs. Abstraktion?
- ❓ Skalierungsgesetze für 4D-Netze?

Siehe: `docs/99-appendix/open-questions.md`

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Für Korrekturen/Ergänzungen: GitHub Issues*
