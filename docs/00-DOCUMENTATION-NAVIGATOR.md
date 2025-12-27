# 4D Neural Cognition - Dokumentations-Navigator

## 🎯 Schnelleinstieg nach Zielgruppe

### Für Forscher (Neurowissenschaft / KI)

**Start hier:**
1. [`00-meta/vision.md`](00-meta/vision.md) - Forschungsziel & Abgrenzung
2. [`00-meta/research-scope.md`](00-meta/research-scope.md) - Was wird/wird nicht untersucht
3. [`01-overview/assumptions.md`](01-overview/assumptions.md) - Grundannahmen
4. [`02-theoretical-foundation/limitations.md`](02-theoretical-foundation/limitations.md) - Bewusste Vereinfachungen
5. [`99-appendix/open-questions.md`](99-appendix/open-questions.md) - Offene Fragen

**Dann weiter zu:**
- Hypothesen: [`SCIENTIFIC_HYPOTHESES.md`](../SCIENTIFIC_HYPOTHESES.md)
- Experimente: `06-experiments/`
- Vergleiche: [`01-overview/prior-art.md`](01-overview/prior-art.md)

---

### Für Entwickler / Contributors

**Start hier:**
1. [`00-meta/contribution-model.md`](00-meta/contribution-model.md) - Wie beitragen?
2. [`00-meta/status.md`](00-meta/status.md) - Was ist stabil/experimentell?
3. [`05-implementation/`](05-implementation/) - Technische Umsetzung
4. [`developer-guide/`](developer-guide/) - Coding Standards

**Dann weiter zu:**
- API: [`api/API.md`](api/API.md)
- Architektur: `03-neural-architecture/`
- Entscheidungen: `07-decisions/` (ADRs)

---

### Für Anwender / Studierende

**Start hier:**
1. [`01-overview/index.md`](01-overview/index.md) - Projektübersicht
2. [`01-overview/glossary.md`](01-overview/glossary.md) - Begriffe
3. [`user-guide/INSTALLATION.md`](user-guide/INSTALLATION.md) - Installation
4. [`tutorials/GETTING_STARTED.md`](tutorials/GETTING_STARTED.md) - Erste Schritte

**Dann weiter zu:**
- Tutorials: `tutorials/`
- Beispiele: `../examples/`
- FAQ: `user-guide/FAQ.md`

---

## 📂 Dokumentations-Struktur (Neue Organisation)

### 00-meta/ - Projektsteuerung

**Zweck:** Governance, Roadmap, Status

| Dokument | Beschreibung | Zielgruppe |
|----------|--------------|------------|
| [`vision.md`](00-meta/vision.md) | Forschungsziel, Nicht-Ziele, wissenschaftliche Positionierung | Alle |
| [`research-scope.md`](00-meta/research-scope.md) | Was wird/wird nicht untersucht, faire Vergleiche | Forscher |
| [`roadmap.md`](00-meta/roadmap.md) | Entwicklungs-Roadmap, Publikationsstrategie | Forscher, Contributors |
| [`status.md`](00-meta/status.md) | Komponenten-Reife (stabil/experimentell/deprecated) | Entwickler |
| [`contribution-model.md`](00-meta/contribution-model.md) | Wissenschaftliche Zusammenarbeit, Beitrag-Richtlinien | Contributors |

**Lesedauer:** 30-45 Min. für alle Dokumente

---

### 01-overview/ - Einordnung

**Zweck:** Projektverständnis, Begriffe, Abgrenzung

| Dokument | Beschreibung | Zielgruppe |
|----------|--------------|------------|
| [`index.md`](01-overview/index.md) | Umfassende Projektübersicht, Quick Start | Alle (START HIER) |
| [`glossary.md`](01-overview/glossary.md) | Neuro-, KI- und Systembegriffe | Alle |
| [`prior-art.md`](01-overview/prior-art.md) | Abgrenzung zu Deep Learning, SNNs, GNNs, etc. | Forscher |
| [`assumptions.md`](01-overview/assumptions.md) | 15 explizite Grundannahmen mit Validierungsplan | Forscher (WICHTIG) |

**Lesedauer:** 60 Min. (komplett)

---

### 02-theoretical-foundation/ - Theorie & Inspiration

**Zweck:** Biologische Bezüge, kognitive Prinzipien

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `neuroscience-basis.md` | Biologische Referenzpunkte | 🔄 In Arbeit |
| `cognitive-principles.md` | Annahmen über Kognition | 🔄 In Arbeit |
| `spatial-intelligence.md` | Continuous Spatial Intelligence Paradigma | 🔄 In Arbeit |
| `time-and-dynamics.md` | Rolle der w-Dimension (Zeit/Hierarchie) | 🔄 In Arbeit |
| [`limitations.md`](02-theoretical-foundation/limitations.md) | **15 bewusste Abweichungen von Biologie** | ✅ Fertig (WICHTIG) |

**Lesedauer:** ~60 Min. (wenn komplett)

---

### 03-neural-architecture/ - Formales Modell

**Zweck:** Mathematisch präzise Modellbeschreibung

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `lattice-structure.md` | 4D neuronales Gitter, Koordinatensystem | 🔄 Migration |
| `neuron-model.md` | LIF, Izhikevich, Zustand, Alter | 🔄 Migration |
| `synapse-model.md` | Verbindungen, Plastizität, Delays | 🔄 Migration |
| `neuromodulation.md` | Modulatoren & globale Effekte | 🔄 Migration |
| `life-cycle.md` | Aging, Death, Reproduction | 🔄 Migration |
| `stability.md` | Erhalt kohärenter Aktivität | 🔄 Migration |

**Referenz:** Bestehende Docs: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`MATHEMATICAL_MODEL.md`](MATHEMATICAL_MODEL.md)

---

### 04-dynamics-and-learning/ - Lernen & Emergenz

**Zweck:** Dynamik, Plastizität, emergentes Verhalten

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `activity-dynamics.md` | Aktivitätsausbreitung, Oszillationen | 🔄 Migration |
| `learning-rules.md` | Hebbian, STDP, lokal vs. global | 🔄 Migration |
| `adaptation.md` | Langsame vs. schnelle Prozesse | 🔄 Migration |
| [`emergence.md`](04-dynamics-and-learning/emergence.md) | **7 operationalisierte Emergenz-Kriterien** | ✅ Fertig (WICHTIG) |
| `failure-modes.md` | Kollaps, Drift, Degeneration | 🔄 Geplant |

**Referenz:** [`LEARNING_SYSTEMS.md`](LEARNING_SYSTEMS.md)

---

### 05-implementation/ - Technische Umsetzung

**Zweck:** Code-Architektur, Performance

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `system-overview.md` | Komponenten-Übersicht | 🔄 Migration |
| `data-representation.md` | JSON, HDF5, Serialisierung | 🔄 Migration |
| `compute-model.md` | Parallelisierung, Skalierung, GPU | 🔄 Migration |
| `hardware-assumptions.md` | CPU/GPU/neuromorph | 🔄 Migration |
| `performance-notes.md` | Bottlenecks, Optimierungsstrategien | 🔄 Migration |

**Referenz:** [`ARCHITECTURE.md`](ARCHITECTURE.md), [`PERFORMANCE_OPTIMIZATION.md`](PERFORMANCE_OPTIMIZATION.md)

---

### 06-experiments/ - Forschung & Evaluation

**Zweck:** Experimentelle Validierung

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `experimental-setup.md` | Reproduzierbare Experimente | 🔄 Migration |
| `benchmarks.md` | Standardisierte Aufgaben | 🔄 Migration |
| `metrics.md` | Was gilt als Erfolg? (Metriken) | 🔄 Migration |
| `reproducibility.md` | Seeds, Configs, Provenance | 🔄 Migration |
| `results-log.md` | Experiment-Ergebnisse | 🔄 Geplant |

**Referenz:** [`BENCHMARK_SUITE.md`](BENCHMARK_SUITE.md), [`SCIENTIFIC_HYPOTHESES.md`](SCIENTIFIC_HYPOTHESES.md)

---

### 07-decisions/ - Architecture Decision Records

**Zweck:** Design-Entscheidungen mit Begründung

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| [`adr-template.md`](07-decisions/adr-template.md) | Vorlage für ADRs | ✅ Fertig |
| [`adr-0001-4d-lattice.md`](07-decisions/adr-0001-4d-lattice.md) | **Warum 4D statt 3D/Graph?** | ✅ Fertig |
| `adr-0002-aging-model.md` | Warum Zell-Lebenszyklus? | 🔄 Geplant |
| [`adr-0003-learning-paradigm.md`](07-decisions/adr-0003-learning-paradigm.md) | **Warum lokale Regeln statt Backprop?** | ✅ Fertig |

**Format:** Kontext, Optionen, Entscheidung, Konsequenzen

---

### 99-appendix/ - Anhang

**Zweck:** Referenzen, Vergleiche, offene Fragen

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| `references.md` | Papers, Bücher, Preprints | 🔄 Migration |
| `comparisons.md` | Systematische Vergleiche mit anderen Modellen | 🔄 Migration |
| [`open-questions.md`](99-appendix/open-questions.md) | **20 explizit ungelöste Probleme** | ✅ Fertig (WICHTIG) |

**Referenz:** [`literature/review.md`](literature/review.md), [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md)

---

## 📚 Bestehende Dokumentation (Legacy)

Diese Dokumente existieren parallel zur neuen Struktur und werden schrittweise migriert:

### Haupt-Dokumente

- [`README.md`](../README.md) - Projekt-Übersicht (wird aktualisiert)
- [`VISION.md`](../VISION.md) - Vision (→ migriert zu `00-meta/vision.md`)
- [`DOCUMENTATION.md`](../DOCUMENTATION.md) - Doku-Index (wird aktualisiert)
- [`CHANGELOG.md`](../CHANGELOG.md) - Versions-Historie
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - Contribution Guide

### Technische Docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - Architektur (→ wird auf `03-neural-architecture/` verteilt)
- [`MATHEMATICAL_MODEL.md`](MATHEMATICAL_MODEL.md) - Math-Modelle (→ `03-neural-architecture/`)
- [`ALGORITHMS.md`](ALGORITHMS.md) - Algorithmen
- [`API_SPECIFICATION.md`](API_SPECIFICATION.md) - API
- [`PERFORMANCE_OPTIMIZATION.md`](PERFORMANCE_OPTIMIZATION.md) - Performance (→ `05-implementation/`)

### Wissenschaftliche Docs

- [`SCIENTIFIC_HYPOTHESES.md`](SCIENTIFIC_HYPOTHESES.md) - 10 Hypothesen
- [`SCIENTIFIC_VALIDATION.md`](SCIENTIFIC_VALIDATION.md) - Validierung
- [`BENCHMARK_SUITE.md`](BENCHMARK_SUITE.md) - Benchmarks (→ `06-experiments/`)
- [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) - Vergleiche (→ `99-appendix/comparisons.md`)

### Spezial-Dokumente

- [`ETHICAL_FRAMEWORK.md`](ETHICAL_FRAMEWORK.md) - Ethik
- [`COLLABORATIVE_RESEARCH.md`](COLLABORATIVE_RESEARCH.md) - Forschungskooperationen
- [`NEUROMORPHIC_HARDWARE_STRATEGY.md`](NEUROMORPHIC_HARDWARE_STRATEGY.md) - Hardware
- [`GPU_ACCELERATION_ROADMAP.md`](GPU_ACCELERATION_ROADMAP.md) - GPU

---

## 🗺️ Lese-Pfade für verschiedene Ziele

### Pfad 1: "Ich will verstehen, worum es geht" (30 Min.)

1. [`01-overview/index.md`](01-overview/index.md) - Projektübersicht
2. [`00-meta/vision.md`](00-meta/vision.md) - Forschungsziel
3. [`01-overview/glossary.md`](01-overview/glossary.md) - Begriffe
4. [`01-overview/prior-art.md`](01-overview/prior-art.md) - Abgrenzung

---

### Pfad 2: "Ich will wissenschaftlich evaluieren" (90 Min.)

1. [`00-meta/research-scope.md`](00-meta/research-scope.md) - Was wird untersucht?
2. [`01-overview/assumptions.md`](01-overview/assumptions.md) - Annahmen
3. [`02-theoretical-foundation/limitations.md`](02-theoretical-foundation/limitations.md) - Limitations
4. [`04-dynamics-and-learning/emergence.md`](04-dynamics-and-learning/emergence.md) - Emergenz-Kriterien
5. [`99-appendix/open-questions.md`](99-appendix/open-questions.md) - Offene Fragen
6. [`SCIENTIFIC_HYPOTHESES.md`](SCIENTIFIC_HYPOTHESES.md) - Hypothesen

---

### Pfad 3: "Ich will beitragen" (60 Min.)

1. [`00-meta/contribution-model.md`](00-meta/contribution-model.md) - Wie beitragen?
2. [`00-meta/status.md`](00-meta/status.md) - Was ist stabil?
3. [`07-decisions/`](07-decisions/) - Design-Entscheidungen
4. [`developer-guide/coding-standards.md`](developer-guide/coding-standards.md) - Code-Standards

---

### Pfad 4: "Ich will es nutzen" (45 Min.)

1. [`01-overview/index.md`](01-overview/index.md) - Übersicht
2. [`user-guide/INSTALLATION.md`](user-guide/INSTALLATION.md) - Installation
3. [`tutorials/GETTING_STARTED.md`](tutorials/GETTING_STARTED.md) - Erste Schritte
4. [`api/API.md`](api/API.md) - API-Referenz

---

## 🔍 Wichtigste Dokumente (Must-Read)

### Für wissenschaftliche Validierung

1. **[`01-overview/assumptions.md`](01-overview/assumptions.md)** - Was wird angenommen?
2. **[`02-theoretical-foundation/limitations.md`](02-theoretical-foundation/limitations.md)** - Was fehlt?
3. **[`04-dynamics-and-learning/emergence.md`](04-dynamics-and-learning/emergence.md)** - Wie wird Emergenz gemessen?
4. **[`99-appendix/open-questions.md`](99-appendix/open-questions.md)** - Was ist unklar?

### Für technisches Verständnis

1. **[`07-decisions/adr-0001-4d-lattice.md`](07-decisions/adr-0001-4d-lattice.md)** - Warum 4D?
2. **[`07-decisions/adr-0003-learning-paradigm.md`](07-decisions/adr-0003-learning-paradigm.md)** - Warum lokale Regeln?
3. **[`ARCHITECTURE.md`](ARCHITECTURE.md)** - System-Architektur

---

## 📊 Status-Übersicht

| Sektion | Fertigstellung | Nächste Schritte |
|---------|----------------|------------------|
| **00-meta** | ✅ 100% | Periodische Updates |
| **01-overview** | ✅ 100% | Feedback einarbeiten |
| **02-theoretical-foundation** | 🟡 20% | Migration bestehender Inhalte |
| **03-neural-architecture** | 🔴 0% | Migration + Reorganisation |
| **04-dynamics-and-learning** | 🟡 20% | Migration LEARNING_SYSTEMS.md |
| **05-implementation** | 🔴 0% | Migration ARCHITECTURE.md, PERFORMANCE |
| **06-experiments** | 🔴 0% | Migration BENCHMARK_SUITE.md |
| **07-decisions** | 🟡 60% | ADR-0002 hinzufügen |
| **99-appendix** | 🟡 33% | References, Comparisons migrieren |

**Gesamt-Fortschritt:** ~40%

---

## 🔄 Migrations-Roadmap

### Phase 1: Kernstruktur (Abgeschlossen ✅)

- [x] Verzeichnisse erstellen
- [x] Kritische Dokumente (Assumptions, Limitations, Emergence, Open Questions)
- [x] ADR-Template + 2 ADRs
- [x] Navigation-Index

### Phase 2: Theorie-Migration (In Arbeit 🔄)

- [ ] `02-theoretical-foundation/` vervollständigen
- [ ] `03-neural-architecture/` aus ARCHITECTURE.md + MATHEMATICAL_MODEL.md
- [ ] `04-dynamics-and-learning/` aus LEARNING_SYSTEMS.md

### Phase 3: Implementierung & Experimente (Geplant 📅)

- [ ] `05-implementation/` aus ARCHITECTURE.md + PERFORMANCE
- [ ] `06-experiments/` aus BENCHMARK_SUITE.md + HYPOTHESES

### Phase 4: Finalisierung (Geplant 📅)

- [ ] README.md aktualisieren
- [ ] DOCUMENTATION.md aktualisieren
- [ ] Cross-References prüfen
- [ ] Vollständigkeits-Check

---

## 💡 Verwendungshinweise

### Für Autoren

- Neue Dokumente: Entsprechende Sektion wählen
- ADRs für wichtige Entscheidungen
- Immer: Assumptions, Limitations, Open Questions prüfen

### Für Reviewer

- Checkliste: Assumptions dokumentiert? Limitations erklärt? Emergenz-Kriterien definiert?
- Cross-References korrekt?

### Für Leser

- Start: `01-overview/index.md`
- Kritisches Lesen: Assumptions + Limitations + Open Questions
- Navigation: Dieses Dokument als Referenz

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0 der neuen Dokumentationsstruktur*  
*Feedback: GitHub Issues*
