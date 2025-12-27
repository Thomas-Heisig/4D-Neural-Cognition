# Limitations - Bewusste Abweichungen von biologischer Realität

## Zweck

Dieses Dokument dokumentiert **explizit**, welche Aspekte biologischer Neuronen und Gehirne **bewusst vereinfacht oder ausgelassen** wurden. Dies ist essentiell für wissenschaftliche Redlichkeit und verhindert falsche Erwartungen.

---

## ⚠️ Grundsätzliche Klarstellung

**4D Neural Cognition ist KEINE biologische Gehirnsimulation.**

Es ist ein **abstraktes Modell**, inspiriert von neurobiologischen Prinzipien, aber mit bewussten Vereinfachungen für:
- Rechenkomplexität
- Konzeptuelle Klarheit
- Fokus auf spezifische Forschungsfragen

---

## 🧠 Neuronale Morphologie

### L1: Punkt-Neuronen statt morphologischer Modelle

**Biologische Realität:**
- Dendriten mit komplexer Verzweigung
- Räumlich verteilte Synapsen (proximal vs. distal)
- Dendritische Spikes und lokale Computation
- Axon-Morphologie beeinflusst Leitgeschwindigkeit

**Unser Modell:**
- Punkt-Neuronen (keine räumliche Ausdehnung)
- Synapsen als skalare Gewichte
- Zentrale Spannungs-Variable (v_membrane)

**Konsequenz:**
- ❌ Keine dendritische Computation (NMDA-Spikes, etc.)
- ❌ Keine räumliche Filterung durch Dendriten
- ❌ Keine Cable-Theory-Effekte

**Literatur:**  
- Häusser & Mel (2003). "Dendrites: bug or feature?" Current Opinion in Neurobiology.
- London & Häusser (2005). "Dendritic computation." Annual Review of Neuroscience.

---

### L2: Keine Glia-Zellen

**Biologische Realität:**
- Astrozyten regulieren synaptische Transmission
- Oligodendrozyten bilden Myelin (Leitgeschwindigkeit)
- Mikroglia (Immunfunktion, Synaptic Pruning)
- Glia-Neuron-Verhältnis ~1:1 im Kortex

**Unser Modell:**
- Nur Neuronen
- Keine Glia-Modellierung

**Konsequenz:**
- ❌ Keine Glia-vermittelte synaptische Plastizität
- ❌ Keine Stoffwechsel-Regulation
- ❌ Keine Immunantwort auf Schädigung

---

## 🔗 Synaptische Transmission

### L3: Vereinfachte Synapsen-Dynamik

**Biologische Realität:**
- Neurotransmitter-Freisetzung (Ca²⁺-abhängig)
- Diffusion im synaptischen Spalt
- Rezeptor-Bindung und Desensitisierung
- Wiederaufnahme (Reuptake)
- Short-term Plasticity (Facilitation, Depression)

**Unser Modell:**
- Spike → sofortige Strom-Injektion (nach Delay)
- Gewicht als skalarer Multiplikator
- Keine kurzfristige Dynamik

**Konsequenz:**
- ❌ Keine kurzfristige synaptische Plastizität
- ❌ Keine Transmitter-Depletion
- ❌ Keine Rezeptor-Sättigung

**Implikation:**  
Working Memory Mechanismen (abhängig von Short-term Plasticity) sind limitiert.

---

### L4: Nur AMPA-artige exzitatorische Synapsen

**Biologische Realität:**
- AMPA (schnell, exzitatorisch)
- NMDA (langsam, voltage-dependent, plastisch)
- GABA_A (schnell, inhibitorisch)
- GABA_B (langsam, modulatorisch)
- Metabotropische Rezeptoren

**Unser Modell:**
- Exzitatorisch: positives Gewicht (AMPA-artig)
- Inhibitorisch: negatives Gewicht (GABA-artig)
- Keine Voltage-Dependenz (wie NMDA)

**Konsequenz:**
- ❌ Keine NMDA-abhängige Koinzidenzdetektion
- ❌ Keine metabotropischen Effekte

---

## 🧬 Genetik & Entwicklung

### L5: Keine neuronale Entwicklung

**Biologische Realität:**
- Neurogenese während Embryonalentwicklung
- Migration und Differenzierung
- Synaptogenese und Pruning
- Myelinisierung
- Erfahrungsabhängige Entwicklung

**Unser Modell:**
- Neuronen werden instantan erstellt (bei Init oder Reproduktion)
- Keine Entwicklungs-Phase
- Keine genetische Regulation

**Konsequenz:**
- ❌ Keine Entwicklungsbiologie
- ❌ Keine kritische Perioden
- ❌ Keine epigenetische Regulation

---

### L6: "Reproduktion" ist keine Neurogenese

**Biologische Realität:**
- Erwachsene Neurogenese: Hippocampus, Olfactory Bulb (begrenzt)
- KEINE Reproduktion bestehender Neuronen

**Unser Modell:**
- Aktivitätsabhängige "Reproduktion" mit Mutation
- Abstrakte Analogie zu struktureller Plastizität

**Klarstellung:**
- ⚠️ Dies ist ein **Mechanismus für Netzwerk-Plastizität**, KEINE biologische Neurogenese
- Inspiriert von genetischen Algorithmen, nicht Neurobiologie

---

## 🔬 Biochemie & Signalling

### L7: Keine Second-Messenger-Systeme

**Biologische Realität:**
- cAMP, cGMP, IP3, DAG
- Protein-Kinasen (PKA, PKC, CaMKII)
- Transkriptionsfaktoren (CREB, etc.)
- Genexpression

**Unser Modell:**
- Direkte Plastizitätsregeln (STDP-Gleichung)
- Keine Biochemie

**Konsequenz:**
- ❌ Keine realistische Zeitskalen für late-LTP
- ❌ Keine Protein-Synthese-abhängige Plastizität
- ❌ Keine Genexpression

---

### L8: Keine Neuromodulatoren (im Detail)

**Biologische Realität:**
- Dopamin (Reward, Motivation)
- Serotonin (Mood, Impulskontrolle)
- Noradrenalin (Arousal, Attention)
- Acetylcholin (Learning, Attention)
- Komplexe Rezeptor-Subtypen (D1-D5 für Dopamin)

**Unser Modell:**
- Abstrakte "Modulatoren" (Prototyp vorhanden)
- Globaler Einfluss auf Plastizität/Erregbarkeit
- Keine Rezeptor-Subtypen

**Konsequenz:**
- ⚠️ Vereinfachte Neuromodulation
- ❌ Keine rezeptor-spezifischen Effekte

---

## ⚡ Elektrophysiologie

### L9: Keine Ionenkanal-Dynamik

**Biologische Realität:**
- Hodgkin-Huxley-Kanäle (Na⁺, K⁺, Ca²⁺)
- Voltage-gated, ligand-gated, mechanosensitive Kanäle
- Hunderte Kanal-Subtypen

**Unser Modell:**
- LIF: Einfache Leck-Gleichung
- Izhikevich: Phänomenologisches Modell
- Keine expliziten Kanäle

**Konsequenz:**
- ❌ Keine Channel-Noise
- ❌ Keine pharmakologische Manipulation (Block einzelner Kanäle)
- ✅ Ausreichend für Spike-Timing

---

## 🌐 Netzwerk-Organisation

### L10: 4D-Gitter vs. biologische Anatomie

**Biologische Realität:**
- Kortikale Schichten (L1-L6)
- Kortikale Säulen (orientation, ocular dominance)
- Long-range Projektionen (Thalamus, Hippocampus ↔ Cortex)
- Nicht-euklidische kortikale Geometrie

**Unser Modell:**
- Euklidisches 4D-Gitter
- w-Dimension als Abstraktion (NICHT biologisch)
- Uniform connectivity (distanzbasiert)

**Konsequenz:**
- ❌ Keine layer-spezifische Verarbeitung (L4 → L2/3 → L5)
- ❌ Keine cortical folds (gyri, sulci)
- ✅ Vereinfachtes Modell für experimentelle Kontrolle

---

### L11: Keine spezialisierte Anatomie

**Biologische Realität:**
- Spezifische Strukturen: Hippocampus, Amygdala, Cerebellum, Basal Ganglia
- Jede mit eigener Architektur

**Unser Modell:**
- Uniform 4D-Gitter
- "Areas" durch w-Koordinate unterschieden

**Konsequenz:**
- ❌ Keine architekturspezifischen Funktionen (z.B. Cerebellum Motor-Learning)

---

## 🧠 Kognition & Verhalten

### L12: Keine motorische Kontrolle

**Biologische Realität:**
- Motor Cortex (M1)
- Cerebellum (Koordination)
- Basal Ganglia (Action Selection)
- Spinal Cord (Reflexe)

**Unser Modell:**
- Abstrakte "Motor Output"-Neuronen
- Keine realistische Motorik

**Konsequenz:**
- ❌ Keine Embodiment (kein echter Roboter-Control)
- ⚠️ Motorik ist placeholder

---

### L13: Keine Bewusstseins-Mechanismen

**Biologische Realität (Hypothesen):**
- Thalamocortical Loops
- Global Workspace Theory
- Integrated Information Theory

**Unser Modell:**
- Keine explizite Bewusstseins-Architektur

**Konsequenz:**
- ❌ Keine Behauptung über Bewusstsein oder Subjektivität

---

## 📊 Energetik & Metabolismus

### L14: Keine metabolischen Constraints

**Biologische Realität:**
- ATP-abhängige Ion-Pumpen
- Glucose-Metabolismus
- Blut-Hirn-Schranke
- Energiekosten begrenzen Feuerrate

**Unser Modell:**
- Unbegrenzte "Energie"
- Keine metabolischen Limits

**Konsequenz:**
- ⚠️ Energieeffizienz-Metriken sind abstrakt (Spike-Counts)
- ❌ Keine realistischen metabolischen Constraints

---

## 🔬 Plastizität & Gedächtnis

### L15: Vereinfachte Langzeit-Plastizität

**Biologische Realität:**
- Early LTP (Minuten): Phosphorylierung
- Late LTP (Stunden-Tage): Protein-Synthese
- Strukturelle Plastizität (Tage-Wochen): Spine Growth
- Systemkonsolidierung (Monate): Hippocampus → Cortex

**Unser Modell:**
- STDP als instantane Gewichtsänderung
- Lifecycle als Langzeit-Mechanismus (abstrakt)

**Konsequenz:**
- ❌ Keine realistischen Zeitskalen für Konsolidierung
- ❌ Keine Hippocampus-Cortex-Interaktion

---

## 📋 Zusammenfassung: Vereinfachungen nach Kategorie

| Kategorie | Ausgelassene Features | Impakt auf Modell |
|-----------|----------------------|-------------------|
| **Morphologie** | Dendriten, Axone, Glia | **Hoch** - Lokale Computation fehlt |
| **Synapsen** | Neurotransmitter-Kinetik, NMDA | **Mittel** - Plastizitätsmechanismen vereinfacht |
| **Entwicklung** | Neurogenese, Migration | **Niedrig** - Nicht im Scope |
| **Biochemie** | Second Messengers, Gene | **Mittel** - Langzeit-Plastizität vereinfacht |
| **Anatomie** | Kortikale Schichten, Spezialstrukturen | **Hoch** - 4D ist Abstraktion |
| **Motorik** | Motor Control, Cerebellum | **Niedrig** - Placeholder |
| **Metabolismus** | ATP, Glucose | **Niedrig** - Abstrahiert |

---

## ✅ Was dennoch erfasst wird

Trotz Vereinfachungen bleiben folgende Prinzipien:

1. **Spiking Dynamics** - Zeitliche Präzision
2. **Lokale Plastizität** - Hebbian/STDP
3. **Räumliche Organisation** - 4D-Gitter
4. **Emergenz** - Komplexe Muster aus einfachen Regeln
5. **Adaption** - Lifecycle als Langzeit-Mechanismus

---

## 🎯 Wann sind diese Limitations problematisch?

**Nicht problematisch für:**
- Hypothesen über 4D-Organisation
- Vergleiche mit anderen abstrakten Modellen (ANNs, SNNs)
- Proof-of-Concept für räumliche Intelligenz

**Problematisch für:**
- ❌ Präzise biologische Vorhersagen
- ❌ Medizinische Anwendungen
- ❌ Detaillierte Neurobiologie-Simulationen

---

## 📖 Transparenz-Verpflichtung

**Wir verpflichten uns:**
- ✅ Alle Vereinfachungen zu dokumentieren
- ✅ Keine impliziten biologischen Behauptungen
- ✅ Bei Publikationen: explizite Limitation-Sektion
- ✅ "Biologisch inspiriert" statt "biologisch realistisch"

---

## 🔗 Siehe auch

- `docs/01-overview/assumptions.md` - Was wir annehmen
- `docs/02-theoretical-foundation/neuroscience-basis.md` - Biologische Inspiration
- `docs/99-appendix/references.md` - Literatur zu biologischer Realität

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 1.0*  
*Dieses Dokument ist essentiell für wissenschaftliche Redlichkeit*
