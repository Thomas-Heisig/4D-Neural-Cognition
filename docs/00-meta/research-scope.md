# Research Scope - Was wird untersucht, was nicht

## Zweck dieses Dokuments

Dieses Dokument definiert explizit die **Grenzen** des Forschungsprojekts. Es dient dazu:
- Unrealistische Erwartungen zu vermeiden
- Forschungsfragen präzise einzugrenzen
- Vergleiche mit anderen Ansätzen fair zu gestalten

---

## ✅ Was wird untersucht

### 1. Räumliche Organisation und 4D-Topologie

**Forschungsfrage:** Bietet eine kontinuierliche 4D-Gitterstruktur Vorteile gegenüber klassischen Schichtarchitekturen?

**Untersuchte Aspekte:**
- Kontinuierliche vs. diskrete räumliche Repräsentation
- Selbstorganisation hierarchischer Strukturen in der w-Dimension
- Räumlich organisierte Konnektivität vs. vollvernetzte Schichten
- Emergenz von Funktionsarealen ohne explizite Vorgabe

**Messbare Hypothesen:**
- H1: 4D-Netzwerke zeigen 20% ± 5% bessere Sample-Effizienz bei räumlichen Reasoning-Tasks
- H2: w-Dimension ermöglicht effizientere temporale Hierarchien als RNNs

**Siehe:** `docs/06-experiments/metrics.md`

---

### 2. Lokale Lernregeln

**Forschungsfrage:** Können lokale Plastizitätsregeln ohne Backpropagation zu funktionaler Spezialisierung führen?

**Untersuchte Aspekte:**
- Hebbian Learning und STDP in 4D-Struktur
- Kombination mit Zell-Lebenszyklus
- Emergenz von Funktions-Clustern
- Vergleich mit Gradient-basierten Methoden

**Bewusste Einschränkung:** 
- Kein Anspruch auf Überlegenheit gegenüber Backpropagation
- Fokus auf biologische Plausibilität und lokale Regeln

---

### 3. Zell-Lebenszyklus und Evolution

**Forschungsfrage:** Kann ein Modell mit Neuronen-Alterung, -Tod und -Reproduktion zu stabiler Funktion führen?

**Untersuchte Aspekte:**
- Alterungsbasierte Homöostase
- Aktivitätsabhängige Reproduktion
- Mutation von Neuronparametern
- Langzeitstabilität ohne explizites Training

**Biologische Einordnung:**
- Dies ist KEINE biologische Neurogenese
- Abstraktes Modell für Netzwerk-Plastizität
- Explorativ, nicht validiert

---

### 4. Zeitliche Dimension (w-Koordinate)

**Forschungsfrage:** Kann die w-Dimension als strukturierendes Prinzip für zeitliche Hierarchien dienen?

**Untersuchte Aspekte:**
- w als Abstraktion statt als reale vierte Raumdimension
- Organisation von Kurz-/Langzeit-Mustern
- Vergleich mit rekurrenten Architekturen

**Nicht-Ziel:**
- Keine physikalische Interpretation von w
- Keine Behauptung über "echte" 4D-Raumzeit

---

### 5. Emergente Phänomene

**Forschungsfrage:** Welche messbaren emergenten Eigenschaften entstehen aus lokalen Regeln?

**Untersuchte Aspekte:**
- Oszillationen (Alpha, Beta, Gamma)
- Travelling Waves
- Kritikalität (Branching Parameter λ ≈ 1)
- Small-World-Eigenschaften

**Operationalisierung:** Siehe `docs/04-dynamics-and-learning/emergence.md`

---

## ❌ Was NICHT untersucht wird

### 1. Biologische Realität

**Nicht behauptet:**
- ❌ "Dies simuliert ein echtes Gehirn"
- ❌ "Neuronen verhalten sich wie biologische Neuronen"
- ❌ "Dies ist ein Modell für Neurogenese"

**Klarstellung:**
- 4D-Gitter ist eine Abstraktion, keine biologische Struktur
- Zell-Lebenszyklus ist ein Mechanismus für Netzwerk-Plastizität, keine Neurobiologie
- Biologische Plausibilität ist Inspiration, nicht Ziel

---

### 2. Allgemeine Künstliche Intelligenz (AGI)

**Nicht behauptet:**
- ❌ "Dies ist ein Weg zu AGI"
- ❌ "Dieses System kann allgemeine Intelligenz entwickeln"
- ❌ "4D-Organisation löst das AGI-Problem"

**Klarstellung:**
- Dies ist ein Forschungswerkzeug, kein AGI-System
- Fokus auf spezifische Hypothesen, nicht auf allgemeine Intelligenz
- AGI ist ein langfristiges, ungeklärtes Forschungsfeld

---

### 3. Leistungsversprechen

**Nicht behauptet:**
- ❌ "Besser als Deep Learning"
- ❌ "Effizienter als alle anderen Ansätze"
- ❌ "State-of-the-art auf Standard-Benchmarks"

**Klarstellung:**
- Vergleiche dienen der Charakterisierung, nicht dem Marketing
- Negative Ergebnisse werden veröffentlicht
- Spezifische Stärken und Schwächen werden dokumentiert

---

### 4. Vollständige Gehirnsimulation

**Nicht Ziel:**
- ❌ Simulation aller Neurotransmitter-Systeme
- ❌ Detaillierte Morphologie (Dendriten, Axone)
- ❌ Gliazellen
- ❌ Metabolische Prozesse
- ❌ Genetische Regulation
- ❌ Entwicklungsbiologie

**Bewusste Vereinfachungen:**
- Punkt-Neuronen statt morphologischer Modelle
- Abstrakte Synapsen statt chemischer Kinetik
- Vereinfachte Plastizität

Siehe: `docs/02-theoretical-foundation/limitations.md`

---

### 5. Kommerzielle Anwendungen

**Nicht Fokus:**
- Produktentwicklung
- Marktreife Software
- Support für Produktions-Deployments
- Performance-Optimierung für spezifische Anwendungen

**Klarstellung:**
- Dies ist ein Forschungsprojekt
- Code ist "as-is" verfügbar (MIT Lizenz)
- Keine Gewährleistung für Produktions-Einsatz

---

## 🔬 Forschungsmethodik

### Was zählt als Erfolg

1. **Peer-reviewed Publikationen**
   - Validierte wissenschaftliche Ergebnisse
   - Reproduzierbare Experimente
   - Statistisch signifikante Unterschiede

2. **Negative Ergebnisse**
   - "4D bringt keinen Vorteil für Aufgabe X" ist ein gültiges Ergebnis
   - Transparente Dokumentation von Fehlschlägen
   - Lernen aus gescheiterten Hypothesen

3. **Charakterisierung, nicht Rangordnung**
   - Identifikation von Aufgabenklassen, für die 4D geeignet ist
   - Verständnis der Mechanismen
   - Vergleich mit etablierten Methoden zur Einordnung

### Was zählt NICHT als Erfolg

- ❌ Anekdotische Beobachtungen ohne Statistik
- ❌ Cherry-picking erfolgreicher Experimente
- ❌ Vergleiche ohne kontrollierte Bedingungen
- ❌ Nicht-reproduzierbare Ergebnisse

---

## 📋 Vergleichsrahmen

### Faire Vergleiche

Bei Vergleichen mit anderen Ansätzen wird sichergestellt:

1. **Gleiches Parameterbudget**
   - Gleiche Anzahl lernbarer Parameter
   - Dokumentierte Netzwerkgröße

2. **Gleiche Rechenzeit**
   - Äquivalente Trainingszeit
   - Dokumentierte Hardware

3. **Gleiche Datensätze**
   - Standardisierte Benchmarks
   - Gleiche Train/Test-Splits

4. **Mehrfache Runs**
   - Statistische Signifikanz
   - Konfidenzintervalle
   - Dokumentierte Varianz

### Baseline-Systeme

Vergleiche gegen:
- Klassische ANNs (MLPs, CNNs, RNNs)
- Spiking Neural Networks (NEST, Brian2)
- Reservoir Computing (ESN, LSM)
- Graph Neural Networks

**Nicht:** Unstandardisierte oder unfaire Vergleiche

---

## 🎯 Abgrenzung von verwandten Ansätzen

| Ansatz | Ähnlichkeiten | Unterschiede |
|--------|---------------|--------------|
| **NEST / Brian2** | Spiking Neurons, Plastizität | 4D-Gitter, Zell-Lebenszyklus |
| **Reservoir Computing** | Fixed Connections, lokales Lernen | Strukturierte Topologie, Evolution |
| **Neural Cellular Automata** | Lokale Regeln, Emergenz | 4D-Koordinaten, biologische Inspiration |
| **HyperNEAT** | Topology-basierte Evolution | Spiking Dynamics, lokale Plastizität |

---

## 📖 Offene Forschungsfragen

Bewusst nicht beantwortete Fragen (siehe `docs/99-appendix/open-questions.md`):

1. Optimale Nutzung der w-Dimension für verschiedene Aufgaben
2. Skalierungsgesetze für 4D-Netzwerke (>1M Neuronen)
3. Kombination mit symbolischen Reasoning-Systemen
4. Biologische Validierung der emergenten Dynamiken
5. Transfer auf neuromorphe Hardware

---

## ✅ Zusammenfassung: Scope Statement

**Dieses Projekt untersucht:**
- Ob 4D-räumliche Organisation Vorteile für neuronale Netze bietet
- Wie lokale Lernregeln in strukturierter Topologie funktionieren
- Welche emergenten Phänomene aus diesen Prinzipien entstehen

**Dieses Projekt behauptet NICHT:**
- Biologische Korrektheit
- Überlegenheit gegenüber etablierten Methoden
- Einen Weg zu AGI
- Produktionsreife

**Erfolgskriterium:**
- Wissenschaftlich validierte Charakterisierung von Stärken und Schwächen
- Reproduzierbare Benchmarks
- Transparente Dokumentation

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Siehe auch: `docs/00-meta/vision.md`, `docs/02-theoretical-foundation/assumptions.md`*
