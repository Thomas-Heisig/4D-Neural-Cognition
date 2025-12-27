# ADR-0001: Warum 4D-Lattice statt 3D oder Graph?

## Kontext und Problemstellung

Neuronale Netze benötigen eine Organisationsstruktur. Klassische Deep Learning nutzt diskrete Schichten, biologische Gehirne haben 3D-Anatomie. Wir müssen entscheiden: Welche Topologie nutzen wir für unser neuromorphes Framework?

**Kernfrage:** Ist eine vier-dimensionale Gitterstruktur sinnvoll gegenüber 2D, 3D oder unstrukturierten Graphen?

## Berücksichtigte Optionen

### Option 1: Traditionelle Layer-Architektur (wie Deep Learning)

**Beschreibung:**
- Diskrete Schichten (Input → Hidden → Output)
- Vollvernetzte oder convolutional layers

**Vorteile:**
- ✅ Etabliert, gut verstanden
- ✅ Effiziente Implementierung (Matrix-Multiplikation)
- ✅ Viele Tools verfügbar (PyTorch, TensorFlow)

**Nachteile:**
- ❌ Nicht biologisch plausibel
- ❌ Keine räumliche Kontinuität
- ❌ Hierarchie explizit vorgegeben (nicht emergent)

---

### Option 2: 3D-Gitter (wie biologisches Gehirn)

**Beschreibung:**
- Neuronen in (x, y, z) positioniert
- Räumlich strukturierte Konnektivität

**Vorteile:**
- ✅ Biologisch realistisch
- ✅ Räumliche Organisation
- ✅ Lokale Konnektivität natürlich

**Nachteile:**
- ❌ Keine natürliche Dimension für Hierarchie/Abstraktion
- ❌ Schwierig, funktionale Hierarchien zu modellieren
- ❌ Zeit-als-Dimension schwer zu repräsentieren

---

### Option 3: Unstrukturierter Graph

**Beschreibung:**
- Neuronen ohne räumliche Koordinaten
- Beliebige Topologie (wie GNNs)

**Vorteile:**
- ✅ Maximale Flexibilität
- ✅ Keine geometrischen Constraints

**Nachteile:**
- ❌ Schwer zu interpretieren
- ❌ Keine räumliche Intuition
- ❌ Biologisch implausibel (Gehirn ist räumlich organisiert)

---

### Option 4: 4D-Gitter (x, y, z, w) ← **GEWÄHLT**

**Beschreibung:**
- Vier-dimensionale euklidische Struktur
- w-Dimension als Meta-Koordinate für Abstraktion/Hierarchie

**Vorteile:**
- ✅ Räumliche Organisation (wie 3D)
- ✅ Zusätzliche Dimension für Hierarchie/Zeit
- ✅ Emergente Strukturen möglich
- ✅ Biologisch inspiriert (ohne strikt biologisch zu sein)
- ✅ Forschungsnovum (differenziert von bestehenden Ansätzen)

**Nachteile:**
- ❌ Höhere Komplexität (4D statt 3D)
- ❌ Mehr Memory (potentiell O(N⁴))
- ❌ Keine direkte biologische Entsprechung für w

## Entscheidung

**Gewählte Option:** Option 4 - 4D-Gitter

**Begründung:**

1. **Forschungshypothese:** Die zentrale Hypothese ist, dass eine zusätzliche Dimension nützlich ist für:
   - Hierarchische Organisation (w=0: sensorisch, w=hoch: abstrakt)
   - Zeitliche Strukturierung (w als temporale Meta-Koordinate)
   - Selbstorganisation funktionaler Areale

2. **Differenzierung:** 4D hebt uns von klassischem Deep Learning (2D-Layers) und biologischen Simulationen (3D) ab.

3. **Experimentelle Flexibilität:**
   - 4D kann zu 3D reduziert werden (w=0 für alle)
   - Erlaubt Vergleichsstudien 2D vs. 3D vs. 4D

4. **Biologische Plausibilität:** Zwar nicht strikt biologisch, aber räumliche Organisation ist plausibel (im Gegensatz zu unstrukturierten Graphen).

**Trade-off akzeptiert:**
- Höhere Komplexität und Memory-Anforderungen für Exploration neuartiger Organisationsprinzipien

## Konsequenzen

### Positive Konsequenzen

- ✅ **Novelty:** Differenziert von bestehenden Ansätzen (Deep Learning, SNNs, Graph-Netze)
- ✅ **Hypothesen-Testbarkeit:** w-Dimension ermöglicht spezifische Experimente
- ✅ **Emergenz:** Potential für selbstorganisierte Hierarchien
- ✅ **Flexibilität:** Reduktion zu 2D/3D möglich für Vergleiche

### Negative Konsequenzen

- ❌ **Skalierung:** Potentiell N⁴ Memory (mitigiert durch Sparse Connections)
- ❌ **Interpretation:** w-Dimension ist abstrakt, schwer zu interpretieren
- ❌ **Validierung:** Keine direkten biologischen Daten zum Vergleich

### Risiken & Mitigation

**Risiko 1: w-Dimension ist nutzlos**
- **Mitigation:** Vergleichsexperimente 3D vs. 4D (Hypothese H1)
- **Fallback:** Reduktion zu 3D, falls 4D keinen Vorteil

**Risiko 2: Memory-Explosion**
- **Mitigation:** Sparse Connectivity (Distanz-basiert)
- **Mitigation:** GPU-Implementierung für Skalierung

**Risiko 3: Biologische Kritik ("Keine w-Dimension im Gehirn")**
- **Mitigation:** Explizite Kommunikation: "Abstraktion, nicht Biologie"
- **Dokumentation:** `limitations.md` klärt Abweichungen

## Validierung

**Erfolgskriterien:**

- [ ] **H1 validiert:** 4D zeigt messbare Vorteile vs. 3D auf räumlichen Tasks (siehe `docs/SCIENTIFIC_HYPOTHESES.md`)
- [ ] **w-Spezialisierung:** Funktionale Unterschiede entlang w-Achse nachweisbar
- [ ] **Skalierung machbar:** >100K Neuronen in 4D simulierbar

**Review-Zeitpunkt:**
- Nach Validierung von Hypothese H1 (geplant Q2 2026)
- Falls H1 widerlegt → Diskussion über Reduktion zu 3D

## Alternativen und Verworfenes

**Warum nicht 3D (Option 2)?**
- Limitiert Exploration hierarchischer Organisation
- Schwierig, Abstraktions-Ebenen zu modellieren
- **Kontra-Argument:** 3D wäre biologisch realistischer
- **Antwort:** Unser Fokus ist Architektur-Exploration, nicht Bio-Simulation

**Warum nicht Graph (Option 3)?**
- Verlust räumlicher Intuition
- Biologisch implausibel
- Schwer zu visualisieren
- **Kontra-Argument:** Maximale Flexibilität
- **Antwort:** Räumliche Struktur ist Teil unserer Hypothese

**Warum nicht Layers (Option 1)?**
- Zu nah an klassischem Deep Learning
- Keine Novelty für Forschung
- **Kontra-Argument:** Etabliert, effizient
- **Antwort:** Performance ist nicht primäres Ziel (Exploration ist Ziel)

## Referenzen

**Inspiration:**
- HyperNEAT (Stanley et al., 2009): Hypercube-basierte Topologie
- Cortical Columns: Räumliche Organisation im Kortex

**Literatur:**
- Stanley, K. O., D'Ambrosio, D. B., & Gauci, J. (2009). "A hypercube-based encoding for evolving large-scale neural networks." Artificial Life.
- Mountcastle, V. B. (1997). "The columnar organization of the neocortex." Brain.

**Verwandte Diskussionen:**
- GitHub Issue #XX: "Why not just use 3D?"
- `docs/01-overview/prior-art.md` - Vergleich mit anderen Ansätzen

---

**Status:** ✅ Accepted  
**Datum:** 2025-12-27  
**Autor:** Thomas Heisig  
**Reviewer:** -  
**Supersedes:** -  
**Superseded by:** - (aktiv)

---

## Notizen für zukünftige Reviews

**Wenn w-Dimension keinen Vorteil zeigt:**
- Erwägen: Reduktion zu 3D-Modell
- Alternative: w für andere Zwecke nutzen (Confidence, Modularity)

**Wenn Skalierung unmöglich:**
- Erwägen: Sparse-only Implementierung
- Alternative: Hybrid (4D für kleine Netze, 3D für große)
