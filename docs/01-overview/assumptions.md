# Assumptions - Grundannahmen des 4D Neural Cognition Modells

## Zweck

Dieses Dokument macht **alle fundamentalen Annahmen** des Modells explizit. Dies ist essentiell für:
- Wissenschaftliche Redlichkeit
- Vergleichbarkeit mit anderen Ansätzen
- Kritische Evaluierung
- Falsifizierbarkeit

⚠️ **Wichtig:** Diese Annahmen sind Hypothesen, keine bewiesenen Fakten.

---

## 🌐 Räumliche Organisation

### A1: Kontinuierlicher 4D-Raum als sinnvolle Abstraktion

**Annahme:**  
Neuronale Information kann sinnvoll in einem kontinuierlichen 4D-euklidischen Raum (x, y, z, w) organisiert werden.

**Begründung:**
- Biologisches Gehirn hat 3D-räumliche Organisation
- w-Dimension als Meta-Koordinate für Abstraktion/Hierarchie

**Vereinfachungen:**
- Reales Gehirn ist 3D, nicht 4D
- Euklidischer Raum vs. komplexe kortikale Geometrie
- Kontinuität vs. diskrete Zellpositionen biologisch

**Testbarkeit:**  
Vergleich mit 2D-, 3D-, und unstrukturierten Netzen auf gleichen Tasks.

**Status:** 🟡 Zu validieren (Hypothese H1)

---

### A2: w-Dimension als Hierarchie-Organisator

**Annahme:**  
Die w-Koordinate kann als strukturierendes Prinzip für kognitive Hierarchien dienen (w=0: sensorisch, w=hoch: abstrakt).

**Biologische Inspiration:**
- Kortikale Hierarchien (V1 → V2 → V4 → IT)
- Prefrontal cortex als "höhere" Verarbeitung

**Unterschied zur Biologie:**
- ❌ Keine physikalische w-Achse im Gehirn
- ✅ Abstrakte Repräsentation von funktionaler Hierarchie

**Testbarkeit:**  
Messung funktionaler Spezialisierung entlang w-Achse.

**Status:** 🟡 Hypothetisch

---

### A3: Räumliche Nähe ↔ Funktionale Ähnlichkeit

**Annahme:**  
Neuronen mit ähnlichen Funktionen clustern räumlich in 4D.

**Biologische Parallele:**
- Kortikale Säulen (orientation columns in V1)
- Tonotopische Karten (A1)
- Somatotopische Karten (S1)

**Mechanismus:**
- Lokale Konnektivität bevorzugt nahe Neuronen
- Plastizität verstärkt funktionale Cluster

**Testbarkeit:**  
Aktivitäts-Korrelation vs. räumliche Distanz.

**Status:** 🟡 Zu validieren

---

## 🧠 Neuronale Dynamik

### A4: Punkt-Neuron-Modelle ausreichend

**Annahme:**  
LIF und Izhikevich-Modelle (Punkt-Neuronen) erfassen wesentliche Dynamik für unsere Forschungsfragen.

**Was fehlt (bewusst vereinfacht):**
- ❌ Dendritische Morphologie
- ❌ Räumliche Verteilung von Synapsen am Neuron
- ❌ Backpropagating Action Potentials
- ❌ Calcium-Dynamik in Dendriten
- ❌ Gap Junctions (elektrische Synapsen)

**Rechtfertigung:**
- Fokus auf Netzwerk-Organisation, nicht Neuron-Detail
- Trade-off: Skalierbarkeit vs. biologische Genauigkeit

**Limitation:**  
Dendritische Computation ist wichtig für biologische Neuronen.

**Status:** ✅ Bewusste Vereinfachung

---

### A5: Diskrete Zeitschritte akzeptabel

**Annahme:**  
Simulation mit diskreten Zeitschritten (dt = 1 ms) erfasst relevante Dynamik.

**Biologische Realität:**
- Kontinuierliche Zeitdynamik

**Unsere Wahl:**
- Euler-Integration mit festem dt
- Kompromiss: Genauigkeit vs. Geschwindigkeit

**Wann problematisch:**
- Bei sehr schnellen Prozessen (<1 ms)
- Bei steifen Differentialgleichungen

**Testbarkeit:**  
Vergleich mit kleineren dt (0.1 ms).

**Status:** ✅ Standard in SNN-Forschung

---

## 🔗 Synapsen & Konnektivität

### A6: Synapsen als gewichtete Verzögerungen

**Annahme:**  
Synapsen können als skalare Gewichte + Verzögerung modelliert werden.

**Was fehlt:**
- ❌ Neurotransmitter-Kinetik
- ❌ Rezeptor-Desensitisierung
- ❌ Calcium-abhängige Freisetzung
- ❌ Short-term Plasticity (Facilitation, Depression)

**Implementiert:**
- ✅ Synaptic Delay
- ✅ Weight (positive = excitatory, negative = inhibitory)
- ✅ Long-term Plasticity (STDP)

**Rechtfertigung:**
- Erste-Ordnung-Approximation ausreichend für Netzwerk-Experimente

**Status:** ✅ Standard-Vereinfachung

---

### A7: Räumlich strukturierte Konnektivität

**Annahme:**  
Verbindungswahrscheinlichkeit fällt mit 4D-Distanz ab:  
```
P(connection) = P₀ · exp(-d / λ)
```

**Parameter:**
- P₀: Basis-Wahrscheinlichkeit
- λ: Verbindungs-Längenkonstante
- d: 4D-euklidische Distanz

**Biologische Inspiration:**
- Kortikale Konnektivität ist räumlich strukturiert
- Lokale Connections > Fern-Connections

**Unterschied zur Biologie:**
- Biologisch: Komplexe Projektions-Muster (z.B. V1→V4)
- Hier: Einfaches distanzbasiertes Modell

**Testbarkeit:**  
Vergleich mit random connections.

**Status:** 🟡 Zu validieren (Teil von H1)

---

## 🎓 Lernregeln

### A8: Lokale Plastizität ausreichend

**Annahme:**  
Hebbian Learning und STDP (lokale Regeln) können zu funktionaler Spezialisierung führen, ohne globale Optimierung (Backpropagation).

**Biologische Plausibilität:**
- ✅ Biologisch plausibel (lokale Information)
- ❌ Keine Credit-Assignment wie Backprop

**Trade-off:**
- **Vorteil:** Biologisch realistisch, parallelisierbar
- **Nachteil:** Möglicherweise suboptimal vs. Gradients

**Nicht behauptet:**
- Überlegenheit gegenüber Backprop
- Äquivalente Performance auf allen Tasks

**Testbarkeit:**  
Vergleich Lokale Regeln vs. Backprop-trained ANN.

**Status:** 🟡 Kernhypothese (H3)

---

### A9: STDP-Parameter biologisch inspiriert

**Annahme:**  
STDP-Fenster (τ₊ = 20 ms, τ₋ = 20 ms) entsprechen biologischen Werten.

**Literatur-Referenz:**
- Bi & Poo (1998): τ ≈ 20 ms in hippocampalen Kulturen
- Markram et al. (1997): Ähnliche Werte in Cortex

**Variation:**
- Biologisch: Variiert zwischen Synapsentypen
- Hier: Uniform angewendet

**Testbarkeit:**  
Parametersweep über τ₊, τ₋.

**Status:** ✅ Literatur-basiert

---

## 🧬 Zell-Lebenszyklus

### A10: Alterung als Homöostase-Mechanismus

**Annahme:**  
Neuronale Alterung (health decay) dient als Regulierungsmechanismus für Netzwerk-Stabilität.

**Biologische Inspiration:**
- ❌ **Nicht biologisch realistisch**: Erwachsene Neuronen sterben nicht regelmäßig
- ✅ **Abstrakte Analogie**: Synaptic Pruning, strukturelle Plastizität

**Mechanismus:**
```python
health -= health_decay_per_step
if health <= 0: neuron_dies()
```

**Zweck:**
- Removal inaktiver Neuronen
- Platz schaffen für Reproduktion

**Klarstellung:**  
Dies ist **KEINE** Simulation von biologischer Neurogenese!

**Status:** 🟡 Experimenteller Mechanismus

---

### A11: Reproduktion mit Mutation als Lernmechanismus

**Annahme:**  
Aktivitätsabhängige Reproduktion mit Parameter-Mutation führt zu adaptiven Netzwerken.

**Inspiration:**
- Genetische Algorithmen (nicht Neurobiologie)
- Strukturelle Plastizität (abstrahiert)

**Mechanismus:**
```python
if neuron.recently_spiked and neuron.health > threshold:
    offspring = reproduce_with_mutation(neuron)
```

**Biologische Einordnung:**
- ❌ Erwachsene Neuronen reproduzieren sich nicht
- ✅ Strukturelle Plastizität existiert (Synaptogenese, Axon-Wachstum)

**Zweck:**
- Exploration von Parameterraum
- Langzeit-Adaptation

**Status:** 🟡 Explorativ, nicht bio-validiert

---

## 📊 Emergenz & Messung

### A12: Emergenz ist operationalisierbar

**Annahme:**  
"Emergente kognitive Strukturen" können durch messbare Kriterien definiert werden.

**Kriterien** (siehe `docs/04-dynamics-and-learning/emergence.md`):
1. Räumliche Clusterung (funktionale Areale)
2. Oszillatorische Dynamik (Alpha, Beta, Gamma)
3. Kritikalität (Branching Parameter λ ≈ 1)
4. Small-World-Eigenschaften

**Problem:**
- Definition von "Kognition" ist nicht konsensual
- Abgrenzung zu Artefakten notwendig

**Lösung:**
- Explizite Metriken
- Vergleich mit Null-Modellen
- Statistische Signifikanz

**Status:** 🟡 Framework definiert, Validierung ausstehend

---

### A13: Performance-Metriken sinnvoll

**Annahme:**  
Standard-ML-Metriken (Accuracy, F1) sind für biologisch inspirierte Modelle relevant.

**Problem:**
- Biologisches Gehirn optimiert nicht für Accuracy
- Andere Ziele: Robustheit, Energieeffizienz, Generalisierung

**Unsere Wahl:**
- ML-Metriken für Vergleichbarkeit
- Zusätzliche Metriken: Energieeffizienz, Biologische Plausibilität

**Status:** ✅ Standard-Praxis

---

## 🔧 Implementierung

### A14: Python-Performance ausreichend

**Annahme:**  
Python (mit NumPy) ist performant genug für unsere Forschungsfragen.

**Realität:**
- Langsamer als C++/CUDA
- Gut genug für ~100K Neuronen

**Skalierung:**
- GPU-Backend für größere Netze (in Entwicklung)
- Neuromorphic Hardware für Deployment

**Status:** ✅ Akzeptabler Trade-off (Entwicklungsgeschwindigkeit vs. Performance)

---

### A15: Einfache Sensorik ausreichend

**Annahme:**  
Direkte Mapping von Inputs zu Neuron-Positionen genügt für Proof-of-Concept.

**Was fehlt:**
- ❌ Realistische Retina-Modelle
- ❌ Cochlea-Filterung
- ❌ Rezeptive Felder

**Implementiert:**
- Vereinfachtes Mapping (Pixel → Neuron)

**Rechtfertigung:**
- Fokus auf 4D-Organisation, nicht sensorische Verarbeitung

**Status:** ✅ Bewusste Vereinfachung

---

## ⚠️ Kritische Annahmen-Abhängigkeiten

### Wenn A2 falsch → w-Dimension nutzlos

Wenn w keine sinnvolle Hierarchie-Organisation bietet:
- Reduktion zu 3D-Modell sinnvoll
- Haupthypothese (H1) widerlegt

### Wenn A8 falsch → Backprop notwendig

Wenn lokale Plastizität fundamental unterlegen:
- Hybrid-Ansätze (lokales Pre-Training + Backprop)
- Fokus auf biologische Exploration, nicht ML-Performance

### Wenn A11 falsch → Lifecycle entfernen

Wenn Reproduktion/Mutation keinen Vorteil bringt:
- Simplifikation zu statischen Netzen
- Fokus nur auf Plastizität

---

## 📋 Zusammenfassung: Annahmen-Kategorien

| Kategorie | Kernannnahmen | Status | Kritikalität |
|-----------|---------------|--------|--------------|
| **4D-Raum** | A1, A2, A3 | 🟡 Zu validieren | **Hoch** |
| **Neuronmodelle** | A4, A5 | ✅ Standard | Mittel |
| **Synapsen** | A6, A7 | ✅ Standard | Mittel |
| **Plastizität** | A8, A9 | 🟡 Kernhypothese | **Hoch** |
| **Lifecycle** | A10, A11 | 🟡 Experimentell | Mittel |
| **Messung** | A12, A13 | ✅ Definiert | Mittel |
| **Technisch** | A14, A15 | ✅ Pragmatisch | Niedrig |

---

## 🔬 Validierungsplan

### Priorität 1 (Essenziell)

1. **A2 (w-Dimension):** Experiments mit w-abhängiger Spezialisierung
2. **A3 (Räumliche Clusterung):** Aktivitäts-Korrelations-Analyse
3. **A8 (Lokale Plastizität):** Vergleich mit Backprop-basierten Modellen

### Priorität 2 (Wichtig)

4. **A7 (Strukturierte Konnektivität):** Vergleich mit random graphs
5. **A11 (Reproduktion):** Ablation-Studie (mit/ohne Lifecycle)

### Priorität 3 (Optional)

6. **A5 (Zeitschritte):** Konvergenz-Test mit kleinerem dt
7. **A15 (Sensorik):** Verbesserung falls nötig

---

## 📖 Referenzen zu Annahmen

- **A4 (Neuronmodelle):** Gerstner et al. (2014). "Neuronal Dynamics."
- **A9 (STDP):** Bi & Poo (1998). "Synaptic modifications in cultured hippocampal neurons."
- **A12 (Kritikalität):** Beggs & Plenz (2003). "Neuronal avalanches in neocortical circuits."

Vollständig: `docs/99-appendix/references.md`

---

## ⚖️ Transparenz-Verpflichtung

**Wir verpflichten uns:**
- ✅ Alle Annahmen zu dokumentieren
- ✅ Negative Ergebnisse zu veröffentlichen, wenn Annahmen widerlegt
- ✅ Annahmen bei neuen Erkenntnissen zu revidieren
- ✅ Keine impliziten Annahmen in Publikationen

**Feedback willkommen:**  
GitHub Issues für Kritik an Annahmen

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Essenzielles Dokument für wissenschaftliche Redlichkeit*
