# Open Questions - Bewusst ungelöste Probleme

## Zweck

Dieses Dokument listet **explizit** offene Forschungsfragen auf, die im aktuellen Projekt:
- Nicht beantwortet werden (können)
- Bewusst ausgeklammert sind
- Zukünftiger Forschung vorbehalten bleiben

Dies verhindert:
- Überinterpretation von Ergebnissen
- Falsche Erwartungen
- Implizite Behauptungen

---

## 🎯 Fundamentale Fragen zur 4D-Organisation

### Q1: Wann ist 4D-Organisation vorteilhaft?

**Frage:**  
Für welche Aufgabenklassen bietet eine 4D-Gitterstruktur tatsächlich Vorteile gegenüber 2D/3D-Netzen oder unstrukturierten Graphen?

**Warum offen:**
- Systematische Exploration über Aufgabenklassen fehlt
- Trade-offs zwischen Dimensionen nicht verstanden
- Möglicherweise gibt es keine universellen Vorteile

**Nächste Schritte:**
- Benchmark-Suite über diverse Tasks
- Vergleichsstudien 2D vs. 3D vs. 4D

**Status:** 🔴 Fundamental ungeklärt

---

### Q2: Optimale Nutzung der w-Dimension

**Frage:**  
Was ist die "richtige" Interpretation und Nutzung der w-Dimension?

**Varianten:**
- w als Abstraktion/Hierarchie (aktueller Ansatz)
- w als Zeitachse (Echo State Networks-artig)
- w als Konfidenz/Gewichtung
- w als Modulationsachse
- Hybrid-Ansätze

**Warum offen:**
- Keine klare biologische Entsprechung
- Verschiedene Ansätze nicht systematisch verglichen

**Nächste Schritte:**
- Ablation-Studien
- Vergleich verschiedener w-Interpretationen

**Status:** 🔴 Konzeptuell ungeklärt

---

### Q3: Skalierungsgesetze für 4D-Netze

**Frage:**  
Wie skaliert Performance und Rechenaufwand mit Netzwerkgröße in 4D?

**Spezifische Fragen:**
- Kritische Größe für Emergenz?
- Memory-Anforderungen: O(N) vs. O(N²) vs. O(N⁴)?
- Optimale Gittergröße pro Dimension?
- Sparsity-Strategien?

**Warum offen:**
- Bisher nur bis ~100K Neuronen getestet
- Kein systematisches Scaling-Experiment

**Nächste Schritte:**
- Scaling-Experimente (10K → 1M Neuronen)
- Sparsity-Optimierungen
- GPU/Neuromorphic-Hardware

**Status:** 🟡 Teilweise erforscht (<100K)

---

## 🧠 Biologische Plausibilität

### Q4: Validierung gegen biologische Daten

**Frage:**  
Repliziert das Modell quantitativ biologische Messwerte?

**Zu testen gegen:**
- Multi-Electrode Array (MEA) Daten
- fMRI/EEG Oszillationen
- Spike-Train-Statistiken (ISI, CV, Fano Factor)
- Korrelations-Strukturen

**Warum offen:**
- Keine systematische Validierung gegen echte Daten
- Keine Kooperationen mit Neuro-Labs

**Nächste Schritte:**
- Allen Brain Atlas Data
- Öffentliche MEA-Datensätze
- Kooperation mit experimentellen Gruppen

**Status:** 🔴 Nicht durchgeführt

---

### Q5: Notwendige vs. hinreichende biologische Features

**Frage:**  
Welche biologischen Mechanismen sind **essentiell** für kognitive Funktion?

**Beispiele:**
- Sind Dendriten notwendig? (aktuell: vereinfacht)
- Ist NMDA-Plastizität essentiell? (aktuell: nur AMPA-artig)
- Ist Neuromodulation kritisch? (aktuell: optional)

**Warum offen:**
- Ablation-Studien fehlen
- Trade-off Realismus vs. Skalierbarkeit unklar

**Nächste Schritte:**
- Systematische Ablation-Experimente
- Vergleich mit full-featured SNNs (NEST, Brian2)

**Status:** 🔴 Konzeptuell ungeklärt

---

## 🔬 Lernen & Plastizität

### Q6: Kombination Lokale Plastizität + Globale Optimierung

**Frage:**  
Kann man lokale Lernregeln (STDP) mit globalen Signalen (Backprop, RL) kombinieren?

**Ansätze:**
- Hybrid: STDP für Reservoir, Backprop für Readout
- Globale Modulatoren (Dopamin) als Lern-Signal
- Meta-Learning über lokale Regeln

**Warum offen:**
- Keine systematische Exploration
- Biologische Plausibilität vs. Performance-Trade-off unklar

**Nächste Schritte:**
- Hybrid-Architekturen testen
- Neuromodulation als globales Signal

**Status:** 🟡 Konzept vorhanden, nicht systematisch getestet

---

### Q7: Zell-Lebenszyklus: Nutzen vs. Komplexität

**Frage:**  
Bringt Aging/Reproduction tatsächlich Vorteile, oder ist es unnötige Komplexität?

**Zu testen:**
- Ablation-Studie: Mit vs. ohne Lifecycle
- Langzeit-Stabilität: 100K+ Schritte
- Transfer-Learning: Vergisst Netz ohne Lifecycle?

**Warum offen:**
- Bisher nur Proof-of-Concept
- Keine rigide Evaluation

**Nächste Schritte:**
- Kontrollierte Experimente
- Vergleich Lifecycle vs. Homeostatic Plasticity allein

**Status:** 🟡 Implementiert, nicht validiert

---

## 🌐 Architektur & Topologie

### Q8: 4D vs. andere Topologien

**Frage:**  
Wie schneidet 4D-Gitter ab gegen andere strukturierte Topologien?

**Vergleichskandidaten:**
- 3D-Gitter (warum nicht ausreichend?)
- Hexagonal Grids (biologisch relevanter?)
- Small-World Networks
- Scale-Free Networks
- Hypergraphs

**Warum offen:**
- Systematische Vergleiche fehlen
- Nur 4D getestet

**Nächste Schritte:**
- Benchmark-Suite über Topologien
- Theoretische Analyse (Dimensionality Curse?)

**Status:** 🔴 Nicht verglichen

---

### Q9: Dynamische Topologie vs. Fixed Grid

**Frage:**  
Sollte die 4D-Struktur statisch sein oder selbst evolvieren?

**Alternativen:**
- Fixed Grid (aktuell)
- Growing Grids (wie Self-Organizing Maps)
- Pruning (Reduktion überflüssiger Dimensionen)
- Adaptive Grids (Resolution je nach Bedarf)

**Warum offen:**
- Dynamische Topologie nicht implementiert
- Biologisch: Cortex ist nicht strikt strukturiert

**Nächste Schritte:**
- Prototyp für wachsende Grids
- Vergleich fixed vs. adaptive

**Status:** 🔴 Nicht exploriert

---

## 🤖 Anwendungen & Integration

### Q10: Neuro-Symbolische Integration

**Frage:**  
Wie kann man 4D-Netze mit symbolischem Reasoning kombinieren?

**Ansätze:**
- Logic Tensor Networks
- Neurale Module + Symbolisches Planning
- Knowledge Graphs einbetten

**Warum offen:**
- Nur Proof-of-Concept vorhanden
- Keine rigide Evaluation

**Nächste Schritte:**
- Benchmark-Tasks (CLEVR, bAbI)
- Kooperation mit Neuro-Symbolic-Forschern

**Status:** 🟡 Framework vorhanden, nicht getestet

---

### Q11: Real-World Deployment

**Frage:**  
Kann man 4D-Netze praktisch einsetzen (Robotik, Sensorfusion)?

**Herausforderungen:**
- Real-time Performance
- Sensorische Rauschrobustheit
- Catastrophic Forgetting bei kontinuierlichem Lernen
- Hardware-Constraints

**Warum offen:**
- Bisher nur Simulationen
- Keine Embodiment-Tests

**Nächste Schritte:**
- Simulation-to-Real Transfer
- Kooperation mit Robotik-Gruppen

**Status:** 🔴 Nicht getestet

---

## 🖥️ Hardware & Skalierung

### Q12: Neuromorphic Hardware Mapping

**Frage:**  
Wie mappt man 4D-Gitter effizient auf neuromorphe Hardware (Loihi, SpiNNaker)?

**Herausforderungen:**
- 4D → 2D/3D Hardware-Mesh mapping
- Routing (lange 4D-Distanzen)
- Memory-Constraints
- Fixed-Point Quantisierung

**Warum offen:**
- Kein Zugang zu Hardware bisher
- Mapping-Algorithmen nicht entwickelt

**Nächste Schritte:**
- Simulation von Hardware-Constraints
- Kooperation mit Intel/SpiNNaker-Teams

**Status:** 🔴 Konzept vorhanden, nicht implementiert

---

### Q13: GPU-Optimierung

**Frage:**  
Optimale GPU-Implementierung für 4D-Netze?

**Fragen:**
- Sparse vs. Dense Tensoren?
- Custom CUDA-Kernel vs. High-Level-Frameworks?
- Memory-Layout für 4D-Zugriff?

**Warum offen:**
- Aktuell nur Prototyp (PyTorch/JAX)
- Keine Performance-Optimierung

**Nächste Schritte:**
- Profiling
- Custom CUDA-Kernel für 4D-Convolution

**Status:** 🟡 Prototyp vorhanden

---

## 🧪 Methodologie & Evaluation

### Q14: Faire Vergleiche mit Deep Learning

**Frage:**  
Wie vergleicht man biologisch plausible Modelle fair mit Backprop-basierten Netzen?

**Probleme:**
- Gleiche Anzahl Parameter?
- Gleiche Trainingszeit?
- Gleiche Daten-Effizienz?
- Biologische Plausibilität vs. Performance-Trade-off

**Warum offen:**
- Kein Konsens in der Community
- Verschiedene Metriken möglich

**Nächste Schritte:**
- Multi-Metrik-Evaluation (Accuracy, Energy, Bio-Plausibility)
- Literatur-Review zu Vergleichsmethodik

**Status:** 🔴 Methodisch ungeklärt

---

### Q15: Benchmark-Suite für 4D-Architekturen

**Frage:**  
Welche Tasks sollten in einer Standard-Benchmark-Suite für 4D-Netze sein?

**Kandidaten:**
- Spatial Reasoning (nutzt 4D-Vorteil?)
- Temporal Prediction (nutzt w als Zeit?)
- Multi-Modal Integration
- Continual Learning

**Warum offen:**
- Keine etablierten 4D-Benchmarks
- Aufgaben müssen 4D-spezifisch sein

**Nächste Schritte:**
- Community-Diskussion
- Datensatz-Erstellung (4D-MNIST?)

**Status:** 🟡 Teilweise definiert (Benchmark-Framework vorhanden)

---

## 📊 Theoretische Grundlagen

### Q16: Mathematische Analyse der 4D-Dynamik

**Frage:**  
Gibt es formale Garantien für Stabilität, Konvergenz, Emergenz in 4D-Netzen?

**Fragen:**
- Lyapunov-Stabilität?
- Fixpunkt-Analyse?
- Phasenübergänge?
- Kapazität (wie Hopfield-Netze)?

**Warum offen:**
- Keine mathematische Theorie für 4D-Gitter mit Plastizität + Lifecycle
- Komplexität hoch

**Nächste Schritte:**
- Vereinfachte Modelle analysieren
- Kooperation mit Theoretikern

**Status:** 🔴 Nicht begonnen

---

### Q17: Informationstheoretische Analyse

**Frage:**  
Wie viel Information wird in 4D-Struktur gespeichert/verarbeitet?

**Metriken:**
- Mutual Information (Input → Output)
- Transfer Entropy (zwischen Arealen)
- Komplexität (Tononi's Φ?)

**Warum offen:**
- Rechenaufwand hoch
- Interpretation schwierig

**Nächste Schritte:**
- Pilot-Studien mit kleinen Netzen
- Approximations-Methoden

**Status:** 🟡 Konzept definiert (docs/advanced/INFORMATION_THEORY.md)

---

## 🌍 Community & Zusammenarbeit

### Q18: Standard-Protokolle für Reproduzierbarkeit

**Frage:**  
Wie sichert man vollständige Reproduzierbarkeit in neuromorphen Experimenten?

**Herausforderungen:**
- Stochastische Prozesse (Seeds)
- Hardware-Abhängigkeiten
- Versionierung (Code, Daten, Configs)

**Warum offen:**
- Best Practices noch nicht etabliert

**Nächste Schritte:**
- Reproducibility-Guidelines schreiben
- Zenodo/figshare für Datensätze

**Status:** 🟡 Teilweise adressiert (Config-System, Seeds)

---

### Q19: Multi-Lab-Validierung

**Frage:**  
Können andere Labs unsere Ergebnisse replizieren?

**Warum wichtig:**
- External Validation
- Unbiased Evaluation
- Community-Building

**Warum offen:**
- Noch keine Publikation
- Keine externen Tests

**Nächste Schritte:**
- Paper-Submission
- GitHub-Release mit Anleitung

**Status:** 🔴 Ausstehend

---

## 🔮 Spekulative Fragen

### Q20: Bewusstsein & Subjektivität

**Frage:**  
Könnte ein hinreichend komplexes 4D-Netz "bewusst" werden?

**Klarstellung:**
- ⚠️ **Hochspekulativ**
- Keine aktuelle Forschungsfrage
- Ethische Implikationen unklar

**Warum hier aufgeführt:**
- Transparenz über Grenzen
- Bewusstes Ausklammern

**Status:** 🔴 Nicht im Scope (siehe ETHICAL_FRAMEWORK.md)

---

## 📋 Zusammenfassung nach Priorität

### Hohe Priorität (Kern-Hypothesen)

- Q1: Wann ist 4D vorteilhaft?
- Q2: Optimale Nutzung der w-Dimension
- Q3: Skalierungsgesetze
- Q4: Biologische Validierung

### Mittlere Priorität (Methodologie)

- Q6: Hybrid-Learning
- Q7: Lifecycle-Nutzen
- Q8: Topologie-Vergleiche
- Q14: Faire Vergleiche
- Q15: Benchmark-Suite

### Niedrige Priorität (Langfristig)

- Q9: Dynamische Topologie
- Q10: Neuro-Symbolisch
- Q12: Neuromorphic Hardware
- Q16: Mathematische Theorie

### Außerhalb Scope

- Q20: Bewusstsein (ethisch/philosophisch)

---

## ⚖️ Transparenz-Verpflichtung

**Wir verpflichten uns:**
- ✅ Offene Fragen zu dokumentieren
- ✅ Keine impliziten Lösungsbehauptungen
- ✅ Update bei neuen Erkenntnissen
- ✅ Negative Ergebnisse als Erkenntnisse

**In Publikationen:**
- Explizite "Limitations & Future Work"-Sektion
- Verweis auf diese Liste

---

## 🔄 Update-Prozess

- **Quarterly Review**: Fragen neu bewerten
- **Bei neuen Erkenntnissen**: Liste aktualisieren
- **Community-Input**: GitHub Issues für neue Fragen

**Letztes Update:** Dezember 2025  
**Nächstes Review:** März 2026

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Transparenz über Grenzen ist wissenschaftlich essentiell*
