# Emergence - Operationalisierung emergenten Verhaltens

## Zweck

Dieses Dokument definiert **messbare Kriterien**, wann von "emergenten kognitiven Strukturen" oder "Emergenz" gesprochen werden kann. Dies ist essentiell, um:
- Nachträgliche Interpretation zu vermeiden
- Artefakte von echter Emergenz zu unterscheiden
- Falsifizierbare Behauptungen zu machen

---

## ⚠️ Problem: Emergenz ist oft vage

**Häufige (problematische) Aussagen:**
- "Das Netzwerk entwickelt intelligentes Verhalten"
- "Kognitive Strukturen emergieren"
- "Das System zeigt selbstorganisierte Muster"

**Problem:**
- Was genau bedeutet "intelligent"?
- Wie unterscheidet man Muster von Rauschen?
- Wann ist Selbstorganisation bedeutsam?

**Lösung:**
- Explizite, messbare Kriterien
- Vergleich mit Null-Modellen
- Statistische Signifikanz

---

## 📊 Kriterien für Emergenz

### E1: Räumliche Funktionale Spezialisierung

**Definition:**  
Neuronen mit ähnlicher Funktion clustern räumlich in 4D.

**Messung:**

```python
# 1. Funktionale Ähnlichkeit
similarity(i, j) = correlation(activity_i, activity_j)

# 2. Räumliche Distanz
distance(i, j) = sqrt((x_i - x_j)² + (y_i - y_j)² + ...)

# 3. Metrik
clustering_score = -Σ similarity(i,j) * distance(i,j)
```

**Kriterium für Emergenz:**
- clustering_score > clustering_score_random + 2σ
- σ = Standardabweichung über randomisierte Netze

**Null-Modell:**
- Gleiches Netz mit random-shuffled Neuron-Positionen
- 100 Wiederholungen für Statistik

**Status:** 🟡 Zu testen (Teil von Hypothese H1)

---

### E2: Hierarchische Organisation entlang w-Dimension

**Definition:**  
Verarbeitung wird abstrakter mit höherem w.

**Messung:**

```python
# Abstraktions-Proxy: Rezeptive Feld-Größe
receptive_field_size(w) = mean(RF-size für Neuronen bei w)

# Erwartung
receptive_field_size sollte monoton steigen mit w
```

**Kriterium:**
- Positive Korrelation: r(w, RF-size) > 0.5, p < 0.05
- Vergleich mit w-shuffled Netz

**Alternative Metriken:**
- Zeitkonstanten (höheres w → langsamere Dynamik)
- Selectivity (höheres w → komplexere Features)

**Status:** 🟡 Metric definiert, Experiment ausstehend

---

### E3: Oszillatorische Dynamik (Biologisch Plausibel)

**Definition:**  
Population-Aktivität zeigt charakteristische Oszillationen.

**Frequenzbänder (biologisch):**
- **Gamma (30-100 Hz):** Lokale Verarbeitung
- **Beta (15-30 Hz):** Motor Control, Top-Down
- **Alpha (8-12 Hz):** Resting State
- **Theta (4-8 Hz):** Memory Encoding

**Messung:**

```python
# Power Spectral Density
from scipy.signal import welch
freqs, psd = welch(population_activity, fs=1000)

# Peak Detection
peaks = find_peaks(psd, height=threshold)

# Kriterium: Mindestens 1 signifikanter Peak
```

**Emergenz-Kriterium:**
- Mindestens 1 Peak mit Power > 2x Baseline
- Peak-Frequenz im biologischen Bereich (4-100 Hz)
- Stabil über Zeit (>1 Sekunde Dauer)

**Null-Modell:**
- Rausch-Aktivität (Poisson-Prozess)
- Sollte flaches Spektrum haben

**Status:** 🟡 Teilweise beobachtet, systematische Charakterisierung ausstehend

---

### E4: Kritikalität (Branching Parameter λ ≈ 1)

**Definition:**  
Netzwerk operiert nahe kritischem Zustand zwischen Inaktivität und Epilepsie.

**Messung (Spike Avalanches):**

```python
# 1. Detektiere Avalanches
avalanche = cascade of spikes within temporal window

# 2. Branching Parameter
λ = <n_{t+1}> / <n_t>
# n_t = Anzahl Spikes zum Zeitpunkt t

# 3. Avalanche-Größen-Verteilung
P(size) ~ size^(-α)  # Power-Law
```

**Emergenz-Kriterium:**
- λ ≈ 1.0 ± 0.1 (kritisch)
- Power-Law-Exponent: α ≈ -1.5 (biologische Werte: -1.2 bis -2.0)
- Goodness-of-Fit: R² > 0.9

**Biologische Evidenz:**
- Beggs & Plenz (2003): λ ≈ 1 in cortical slices

**Status:** ✅ Teilweise validiert (λ ≈ 0.95 beobachtet)

**Literatur:**  
Beggs & Plenz (2003). "Neuronal avalanches in neocortical circuits." Journal of Neuroscience.

---

### E5: Small-World Network Eigenschaften

**Definition:**  
Netzwerk kombiniert hohe lokale Clusterung mit kurzen Pfadlängen.

**Metriken:**

```python
# 1. Clustering Coefficient
C = (Anzahl Dreiecke) / (Anzahl verbundener Triplets)

# 2. Average Path Length
L = mean(shortest_path(i, j) for all pairs i,j)

# 3. Small-World-Index
σ = (C / C_random) / (L / L_random)
```

**Emergenz-Kriterium:**
- σ > 1.5 (Small-World)
- C > C_random (hohe lokale Clusterung)
- L ≈ L_random (kurze Pfade trotzdem)

**Biologische Relevanz:**
- Bassett & Bullmore (2006): Kortikale Netze sind Small-World

**Status:** 🟡 Zu testen

---

### E6: Travelling Waves

**Definition:**  
Räumlich propagierende Aktivitätswellen.

**Messung:**

```python
# 1. Cross-Correlation zwischen Positionen
xcorr(position_1, position_2, lag)

# 2. Delay vs. Distance
delay = argmax(xcorr)
velocity = distance / delay

# 3. Wavefront Detection
wavefront = contour of simultaneous activation
```

**Emergenz-Kriterium:**
- Propagation Velocity: 0.1-0.3 m/s (biologische Range)
- Konsistente Richtung über multiple Trials
- Wellenfront kohärent (nicht fragmentiert)

**Biologische Evidenz:**
- Ermentrout & Kleinfeld (2001): Travelling waves in sensory cortex

**Status:** 🟡 Zu testen

---

### E7: Funktionale Konnektivität ≠ Strukturelle Konnektivität

**Definition:**  
Funktionale Verbindungen (Aktivitäts-Korrelation) sind nicht identisch mit strukturellen Verbindungen (Synapsen).

**Messung:**

```python
# Strukturell
structural_conn[i,j] = 1 if synapse(i,j) exists else 0

# Funktional
functional_conn[i,j] = correlation(activity_i, activity_j)

# Vergleich
similarity = overlap(structural_conn, functional_conn > threshold)
```

**Emergenz-Kriterium:**
- Funktionale Konnektivität ist NICHT nur strukturelle Nachbarschaft
- Fernverbindungen trotz fehlender direkter Synapsen
- Similarity < 0.8 (nicht perfekte Überlappung)

**Interpretation:**
- Funktionale Module emergieren aus Interaktionen

**Status:** 🟡 Zu testen

---

## ❌ Was NICHT als Emergenz zählt

### N1: Triviale Konsequenzen der Architektur

**Beispiel:**
- "Neuronen bei w=0 reagieren auf Inputs"
- **Warum nicht Emergenz:** Input wird explizit zu w=0 gemappt

**Regel:**
- Wenn Verhalten direkt aus Architektur-Design folgt → KEINE Emergenz

---

### N2: Zufällige Muster

**Beispiel:**
- "Netzwerk zeigt komplexe räumliche Muster"
- **Problem:** Auch Rauschen hat Muster

**Lösung:**
- Vergleich mit Null-Modell erforderlich
- Statistische Signifikanz (p < 0.05)

---

### N3: Anekdotische Beobachtungen

**Beispiel:**
- "Ich habe einmal ein interessantes Cluster gesehen"

**Problem:**
- Nicht reproduzierbar
- Könnte Zufall sein

**Lösung:**
- Systematische Analyse über multiple Seeds
- Quantitative Metriken

---

### N4: Anthropomorphe Interpretation

**Beispiel:**
- "Das Netzwerk 'versteht' Objekte"

**Problem:**
- "Verstehen" ist nicht definiert

**Lösung:**
- Operationale Definitionen (Klassifikationsgenauigkeit, etc.)

---

## 🧪 Experimenteller Workflow

### Schritt 1: Hypothese formulieren

```
"Funktionale Spezialisierung emergiert in 4D-Gittern"
```

### Schritt 2: Metrik wählen

```
Clustering Score (E1)
```

### Schritt 3: Null-Modell definieren

```
Random-shuffled Neuron-Positionen
```

### Schritt 4: Experiment durchführen

```python
scores = []
for seed in range(100):
    model = create_4d_network(seed)
    model.train(1000 steps)
    score = compute_clustering_score(model)
    scores.append(score)

scores_null = []
for seed in range(100):
    model_null = create_random_network(seed)
    model_null.train(1000 steps)
    score_null = compute_clustering_score(model_null)
    scores_null.append(score_null)
```

### Schritt 5: Statistische Analyse

```python
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(scores, scores_null)

if p_value < 0.05 and mean(scores) > mean(scores_null):
    print("Emergenz nachgewiesen!")
else:
    print("Keine signifikante Emergenz")
```

### Schritt 6: Dokumentation

- Alle Parameter dokumentieren
- Seeds speichern
- Plots generieren
- Ergebnis in `results-log.md`

---

## 📋 Emergenz-Checkliste

Für jede Behauptung über Emergenz:

- [ ] **Explizite Metrik** definiert?
- [ ] **Null-Modell** spezifiziert?
- [ ] **Statistische Signifikanz** (p < 0.05)?
- [ ] **Reproduzierbar** (multiple Seeds)?
- [ ] **Nicht trivial** (aus Architektur ableitbar)?
- [ ] **Dokumentiert** (Config, Seeds, Code)?

---

## 🎯 Zusammenfassung: Emergenz-Kriterien

| Kriterium | Metrik | Schwellenwert | Status |
|-----------|--------|---------------|--------|
| **E1: Funktionale Cluster** | Clustering Score | >random + 2σ | 🟡 Zu testen |
| **E2: Hierarchie (w)** | Korrelation(w, RF-size) | r > 0.5 | 🟡 Zu testen |
| **E3: Oszillationen** | PSD Peaks | Power >2x Baseline | 🟡 Beobachtet |
| **E4: Kritikalität** | Branching λ | λ ≈ 1.0 ± 0.1 | ✅ Teilweise validiert |
| **E5: Small-World** | σ | σ > 1.5 | 🟡 Zu testen |
| **E6: Travelling Waves** | Velocity | 0.1-0.3 m/s | 🟡 Zu testen |
| **E7: Funktionale Conn.** | Similarity | < 0.8 | 🟡 Zu testen |

---

## 📖 Literatur zu Emergenz-Metriken

1. **Kritikalität:**  
   Beggs & Plenz (2003). "Neuronal avalanches in neocortical circuits." Journal of Neuroscience.

2. **Small-World:**  
   Bassett & Bullmore (2006). "Small-world brain networks." The Neuroscientist.

3. **Travelling Waves:**  
   Ermentrout & Kleinfeld (2001). "Traveling electrical waves in cortex." Neuron.

4. **Oszillationen:**  
   Buzsáki & Draguhn (2004). "Neuronal oscillations in cortical networks." Science.

Vollständig: `docs/99-appendix/references.md`

---

## ⚖️ Transparenz-Verpflichtung

**Wir verpflichten uns:**
- ✅ Nur messbare Emergenz-Behauptungen
- ✅ Immer Null-Modell-Vergleich
- ✅ Statistische Signifikanz erforderlich
- ✅ Negative Ergebnisse veröffentlichen

**Vermeiden:**
- ❌ Vage Begriffe ohne Metrik
- ❌ Anthropomorphe Sprache
- ❌ Cherry-Picking interessanter Beispiele

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Essentiell für wissenschaftliche Validität*
