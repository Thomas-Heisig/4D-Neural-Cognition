# ADR-0003: Lokale Lernregeln statt Backpropagation

## Kontext und Problemstellung

Neuronale Netze müssen lernen. Klassisches Deep Learning nutzt Backpropagation – ein globaler, nicht-lokaler Algorithmus. Biologische Gehirne nutzen lokale Plastizitätsregeln (Hebbian, STDP).

**Kernfrage:** Sollten wir Backpropagation nutzen (effizienter) oder lokale Regeln (biologisch plausibel)?

## Berücksichtigte Optionen

### Option 1: Backpropagation (Standard Deep Learning)

**Beschreibung:**
- Gradientenberechnung über gesamtes Netzwerk
- Weight-Update basiert auf Output-Error
- Requires differentiable activation functions

**Vorteile:**
- ✅ Sehr effizient (SOTA Performance)
- ✅ Gut verstanden, viele Tools
- ✅ Garantierte Konvergenz (bei richtiger Tuning)

**Nachteile:**
- ❌ Biologisch implausibel (keine Backwards-Pass im Gehirn)
- ❌ Nicht parallelisierbar (sequenzielle Layer-Updates)
- ❌ Credit Assignment Problem künstlich gelöst
- ❌ Differenziert uns nicht von Deep Learning

---

### Option 2: Hebbian Learning (Lokal, Korrelation-basiert) ← **GEWÄHLT**

**Beschreibung:**
- "Neurons that fire together, wire together"
- Gewichts-Update basiert nur auf Pre/Post-Aktivität
- Keine globalen Signale notwendig

**Vorteile:**
- ✅ Biologisch plausibel
- ✅ Vollständig lokal (parallelisierbar)
- ✅ Keine Supervision notwendig (unsupervised)
- ✅ Differenziert uns von Deep Learning

**Nachteile:**
- ❌ Suboptimal für viele Tasks (vs. Backprop)
- ❌ Keine Credit Assignment
- ❌ Tendenz zu Runaway-Potentiation (mitigiert durch Decay)

---

### Option 3: STDP (Spike-Timing Dependent Plasticity) ← **GEWÄHLT**

**Beschreibung:**
- Zeitabhängige Version von Hebbian
- Pre-Post-Timing entscheidet über LTP/LTD
- Lokale Regel, biologisch validiert

**Vorteile:**
- ✅ Biologisch realistisch (experimentell nachgewiesen)
- ✅ Zeitliche Präzision (kausale Beziehungen)
- ✅ Lokal

**Nachteile:**
- ❌ Komplex zu parametrisieren (τ₊, τ₋, A₊, A₋)
- ❌ Langsamer als Hebbian (zeitliche Fenster)
- ❌ Weniger gut verstanden als Backprop

---

### Option 4: Hybrid (Lokale Regeln + Globale Modulation)

**Beschreibung:**
- Basis: Hebbian/STDP (lokal)
- Zusätzlich: Globale Reward-Signale (Dopamin-artig)

**Vorteile:**
- ✅ Biologisch plausibel (Neuromodulation existiert)
- ✅ Credit Assignment möglich (über Dopamin)
- ✅ Best of both worlds?

**Nachteile:**
- ❌ Komplexität
- ❌ Noch nicht implementiert (Prototyp vorhanden)

## Entscheidung

**Gewählte Optionen:** Option 2 + 3 (Hebbian + STDP)  
**Plus (optional):** Option 4 (Hybrid mit Neuromodulation) als zukünftige Erweiterung

**Begründung:**

1. **Biologische Plausibilität:**
   - Zentrales Projektziel ist biologisch inspiriertes Lernen
   - Backprop wäre Widerspruch zu diesem Ziel

2. **Differenzierung:**
   - Lokale Regeln unterscheiden uns von klassischem Deep Learning
   - Forschungsfrage: "Wie weit kommt man mit lokalen Regeln?"

3. **Parallelisierung:**
   - Lokale Regeln sind intrinsisch parallelisierbar
   - Wichtig für neuromorphe Hardware

4. **Wissenschaftliche Ehrlichkeit:**
   - Wir behaupten NICHT, besser als Backprop zu sein
   - Fokus: Charakterisierung lokaler Regeln in 4D-Kontext

**Trade-off akzeptiert:**
- Performance-Einbußen auf Standard-Benchmarks für biologische Plausibilität

## Konsequenzen

### Positive Konsequenzen

- ✅ **Biologisch plausibel:** Näher an echtem Gehirn
- ✅ **Lokal:** Jede Synapse kann unabhängig lernen
- ✅ **Parallelisierbar:** Wichtig für Skalierung
- ✅ **Forschungsnovum:** Lokale Regeln in 4D-Struktur

### Negative Konsequenzen

- ❌ **Performance:** Wahrscheinlich schlechter als Backprop auf Standard-Tasks
- ❌ **Training:** Langsamer, weniger vorhersagbar
- ❌ **Debugging:** Schwieriger (keine Gradients zum Debuggen)

### Risiken & Mitigation

**Risiko 1: Zu schlechte Performance (Netzwerk lernt nichts)**
- **Mitigation:** Homeostatic Plasticity (stabilisiert)
- **Mitigation:** Weight Decay (verhindert Runaway)
- **Mitigation:** Sorgfältiges Tuning (Learning Rates, etc.)
- **Fallback:** Hybrid-Ansatz (STDP + globales Signal)

**Risiko 2: Nicht vergleichbar mit State-of-the-Art**
- **Mitigation:** Explizite Kommunikation: "Nicht SOTA-Ziel"
- **Mitigation:** Vergleich mit anderen biologisch plausiblen Modellen (NEST, Brian2)

**Risiko 3: Community-Ablehnung ("Warum nicht Backprop?")**
- **Mitigation:** Klare Positionierung als biologisches Forschungsprojekt
- **Dokumentation:** `research-scope.md` erklärt Nicht-Ziele

## Validierung

**Erfolgskriterien:**

- [ ] **Basisfunktionalität:** Netzwerk lernt einfache Assoziationen (Hebbian)
- [ ] **Zeitliche Kausalität:** STDP zeigt kausale Lerneffekte
- [ ] **Stabilität:** Langzeit-Training ohne Kollaps (>10K Schritte)
- [ ] **Vergleich:** Performance vs. andere lokale Lernmodelle dokumentiert

**Nicht Erfolgskriterium:**
- ❌ Performance vs. Backprop (erwartet schlechter)

**Review-Zeitpunkt:**
- Nach ersten Experimenten (Q1 2026)
- Falls grundlegend nicht funktional → Hybrid-Ansatz erwägen

## Alternativen und Verworfenes

**Warum nicht Backprop (Option 1)?**
- **Widerspruch zur Vision:** Biologisch plausibles Modell
- **Keine Differenzierung:** Zu nah an klassischem Deep Learning
- **Kontra-Argument:** "Aber Backprop funktioniert besser!"
- **Antwort:** Performance ist nicht primäres Ziel. Biologische Plausibilität und Exploration sind Ziel.

**Warum nicht nur Hebbian (Option 2 ohne STDP)?**
- **STDP bietet Vorteile:** Zeitliche Präzision, kausale Beziehungen
- **Biologie:** STDP ist experimentell nachgewiesen
- **Kombination:** Hebbian als Fallback, STDP als präzise Variante

**Warum nicht sofort Hybrid (Option 4)?**
- **Komplexität:** Zuerst einfache Ansätze testen
- **Forschungsstrategie:** Schrittweise Erweiterung
- **Später:** Hybrid-Ansatz als Extension geplant

## Implementierungsdetails

### Hebbian-Parameter

```python
learning_rate = 0.01
weight_bounds = (0.0, 1.0)
decay_rate = 0.001  # Verhindert Runaway-Potentiation
```

### STDP-Parameter

```python
A_plus = 0.01   # LTP-Amplitude
A_minus = 0.012  # LTD-Amplitude (leicht höher für Balance)
tau_plus = 20   # ms (Literatur: Bi & Poo 1998)
tau_minus = 20  # ms
```

### Stabilisierungs-Mechanismen

- **Weight Decay:** Verhindert unbegrenzte Verstärkung
- **Weight Clipping:** Hard Bounds
- **Homeostatic Plasticity:** Langfristige Stabilität (optional)

## Referenzen

**Biologische Grundlagen:**
- Hebb, D. O. (1949). "The Organization of Behavior."
- Bi, G., & Poo, M. (1998). "Synaptic modifications in cultured hippocampal neurons." Journal of Neuroscience.

**Computationale Modelle:**
- Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity." Nature Neuroscience.

**Vergleichsstudien:**
- Tavanaei, A., et al. (2019). "Deep learning in spiking neural networks." Neural Networks.

**Verwandte Dokumente:**
- `docs/01-overview/assumptions.md` - Annahme A8: Lokale Plastizität
- `docs/04-dynamics-and-learning/learning-rules.md` - Detaillierte Beschreibungen
- `docs/02-theoretical-foundation/limitations.md` - Was fehlt (kein Backprop)

---

**Status:** ✅ Accepted  
**Datum:** 2025-12-27  
**Autor:** Thomas Heisig  
**Reviewer:** -  
**Supersedes:** -  
**Superseded by:** - (aktiv)

---

## Zukünftige Erweiterungen

**Geplant (Q2 2026):**
- Neuromodulation (Dopamin-artiges Signal für Reward)
- Hybrid-Ansatz: STDP + globales Learning-Signal

**Nicht geplant:**
- Backpropagation im Kern-Modell
- Gradient-basierte Optimierung

**Offene Forschungsfrage:**
- Kann man lokale Regeln + Meta-Learning kombinieren? (siehe `docs/99-appendix/open-questions.md` Q6)
