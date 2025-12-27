# Contribution Model - Forschungsorientierte Zusammenarbeit

## Zweck

Dieses Dokument beschreibt, wie Beiträge zum 4D Neural Cognition Projekt im Kontext eines **Forschungsprojekts** strukturiert sind. Es unterscheidet sich von typischen Open-Source-Projekten durch den Fokus auf wissenschaftliche Validierung.

---

## 🎯 Arten von Beiträgen

### 1. Wissenschaftliche Beiträge

**Hypothesen-Tests**
- Neue testbare Hypothesen formulieren
- Bestehende Hypothesen experimentell validieren
- Negative Ergebnisse sind wertvoll!

**Experimente & Benchmarks**
- Neue Benchmark-Tasks entwickeln
- Vergleichsstudien mit anderen Modellen
- Reproduktion bestehender Experimente

**Theoretische Arbeiten**
- Mathematische Analyse von 4D-Eigenschaften
- Formale Modellierung emergenter Phänomene
- Verbindungen zu Neurowissenschaft oder ML-Theorie

**Anforderungen:**
- Reproduzierbare Methodik
- Statistische Signifikanz (wo anwendbar)
- Transparente Dokumentation
- Negative Ergebnisse willkommen

**Anerkennung:**
- Co-Autorschaft bei Paper-Beiträgen
- Citation in wissenschaftlichen Publikationen
- Nennung in CHANGELOG.md

---

### 2. Code-Beiträge

**Kern-Komponenten (hohe Anforderungen)**
- Tests erforderlich (80%+ Coverage für neue Features)
- Code-Review durch Maintainer
- Dokumentation (Docstrings, README-Updates)
- API-Stabilität beachten (siehe `docs/00-meta/status.md`)

**Experimentelle Features (moderate Anforderungen)**
- Als "experimentell" markiert in `status.md`
- Tests erwünscht, aber nicht zwingend
- API kann sich ändern
- Dokumentation erwünscht

**Beispiele und Tutorials**
- Willkommen für Lernanwendungen
- Keine strengen Tests erforderlich
- Fokus auf Verständlichkeit

**Best Practices:**
```bash
# 1. Feature Branch erstellen
git checkout -b feature/your-feature-name

# 2. Code schreiben + Tests
# 3. Linting
pylint src/your_module.py
black src/your_module.py

# 4. Tests lokal ausführen
pytest tests/test_your_module.py

# 5. Pull Request erstellen
```

---

### 3. Dokumentations-Beiträge

**Hoch willkommen:**
- Erklärung komplexer Konzepte
- Tutorials für Anfänger
- API-Dokumentation vervollständigen
- Korrekturen von Fehlern oder Unklarheiten
- Übersetzungen (Englisch ↔ Deutsch)

**Struktur beachten:**
- Siehe neue Ordnerstruktur (`docs/00-meta/` bis `docs/99-appendix/`)
- Wissenschaftliche Sprache, keine Marketing-Formulierungen
- Annahmen explizit dokumentieren

**Format:**
- Markdown
- Klare Überschriften
- Code-Beispiele wo sinnvoll
- Referenzen zu Literatur

---

### 4. Datensätze & Benchmarks

**Neue Benchmarks:**
- Klare Aufgabenbeschreibung
- Baseline-Ergebnisse
- Reproduzierbare Durchführung
- Offene Lizenz (MIT, CC-BY, etc.)

**4D-spezifische Datensätze:**
- Besonders wertvoll!
- Dokumentation der Generierung
- Train/Test-Splits definiert
- Veröffentlichung als separate Repository möglich

**Anerkennung:**
- Citation als Datensatz-Autor
- DOI-Vergabe bei signifikanten Datensätzen (Zenodo)

---

## 🔬 Wissenschaftliche Kooperationen

### Institutionelle Partnerschaften

**Für Forschungsgruppen:**
1. **Joint Research Projects**
   - Gemeinsame Hypothesentests
   - Co-Authored Papers
   - Geteilte Resourcen (Compute, Hardware)

2. **Student Projects**
   - Bachelor-/Masterarbeiten
   - PhD-Forschung
   - Praktika

3. **Hardware-Zugang**
   - Neuromorphe Hardware-Tests (Loihi, SpiNNaker)
   - GPU-Cluster für Skalierungstests

**Prozess:**
- Kontakt: t_heisig@gmx.de
- Gemeinsame Definition von Zielen
- Dokumentation in `docs/COLLABORATIVE_RESEARCH.md`

---

### Akademische Paper

**Co-Autorschaft-Richtlinien:**

Gemäß ICMJE-Kriterien:
1. Substanzieller Beitrag zu Konzeption oder Datenakquise/Analyse
2. Entwurf oder kritische Revision des Manuskripts
3. Finale Freigabe
4. Verantwortung für Integrität der Arbeit

**Publikationsstrategie:**
- Preprints auf arXiv willkommen
- Peer-Review bevorzugt
- Open Access angestrebt
- Negative Ergebnisse veröffentlichungswürdig

---

## 📋 Contribution-Workflow

### Schritt 1: Issue erstellen/diskutieren

**Vor größeren Beiträgen:**
- GitHub Issue öffnen
- Diskussion mit Maintainer
- Vermeidung von Duplikaten

**Templates:**
- Bug Report
- Feature Request
- Research Hypothesis
- Benchmark Proposal

### Schritt 2: Fork & Branch

```bash
# Fork auf GitHub
# Clone deines Forks
git clone https://github.com/YOUR_USERNAME/4D-Neural-Cognition.git

# Upstream hinzufügen
git remote add upstream https://github.com/Thomas-Heisig/4D-Neural-Cognition.git

# Feature Branch
git checkout -b feature/descriptive-name
```

### Schritt 3: Entwicklung

**Code-Standards:**
- Python 3.8+ kompatibel
- Type Hints erwünscht
- Docstrings (Google Style)
- Black Formatting
- Pylint >8.0/10

**Tests:**
- pytest Framework
- Für Kern-Features: 80%+ Coverage
- Für experimentelle Features: erwünscht

**Dokumentation:**
- README.md-Updates bei API-Änderungen
- Docstrings für alle öffentlichen Funktionen
- Tutorial für neue Features (optional)

### Schritt 4: Pull Request

**PR-Checkliste:**
- [ ] Tests lokal bestanden
- [ ] Code formatiert (black, pylint)
- [ ] Dokumentation aktualisiert
- [ ] CHANGELOG.md-Eintrag (für größere Features)
- [ ] Issue-Referenz im PR

**PR-Template:**
```markdown
## Beschreibung
Kurze Beschreibung der Änderung

## Motivation
Warum ist diese Änderung sinnvoll?

## Art der Änderung
- [ ] Bug Fix
- [ ] Neue Feature
- [ ] Breaking Change
- [ ] Dokumentation
- [ ] Forschungs-Beitrag

## Tests
Wie wurde getestet?

## Checklist
- [ ] Tests geschrieben
- [ ] Dokumentation aktualisiert
- [ ] Linting bestanden
```

### Schritt 5: Review & Merge

**Review-Prozess:**
1. Automatische CI/CD-Checks
2. Code-Review durch Maintainer
3. Eventuelle Änderungswünsche
4. Merge bei Zustimmung

**Merge-Kriterien:**
- CI/CD grün
- Code-Review approval
- Keine Merge-Konflikte
- Tests für neue Features

---

## 🧪 Experimentelle Features

**Kennzeichnung:**
- Status in `docs/00-meta/status.md` als "🟡 Experimentell"
- Warnung in Dokumentation
- API kann sich ändern

**Übergang zu "Stabil":**
- 80%+ Test-Coverage
- Verwendet in mindestens 1 Publikation/Projekt
- API über 3 Monate stabil
- Maintainer-Entscheidung

---

## 📊 Qualitäts-Richtlinien

### Code-Qualität

| Metrik | Kern-Feature | Experimentell | Beispiel |
|--------|--------------|---------------|----------|
| Test-Coverage | >80% | >50% erwünscht | Beliebig |
| Pylint-Score | >8.5/10 | >7.0/10 | >6.0/10 |
| Type Hints | Vollständig | Teilweise | Optional |
| Docstrings | Vollständig | Wichtige Funktionen | Optional |

### Wissenschaftliche Qualität

**Für Experimente:**
- Reproduzierbare Methodik (Seeds, Configs dokumentiert)
- Statistische Signifikanz wo möglich (α = 0.05)
- Negative Ergebnisse dokumentiert
- Vergleich mit Baselines

**Für Hypothesen:**
- Messbare Metriken definiert
- Falsifizierbarkeit sichergestellt
- Literatur-Referenzen

---

## 🏆 Anerkennung

### Contributors-Liste

Alle Beiträge werden in `README.md` und `CHANGELOG.md` anerkannt:

**Kategorien:**
- 🏅 Core Contributors (>10 substantielle PRs)
- 🧪 Research Contributors (Hypothesen, Experimente)
- 📝 Documentation Contributors
- 🐛 Bug Hunters
- 💡 Feature Proposers

### Paper-Autorschaft

**Kriterien für Co-Autorschaft:**
- Substanzielle wissenschaftliche Beiträge
- Code-Beiträge allein: Acknowledgment
- Experimente/Analysen: Co-Autorschaft möglich
- Hypothesen-Formulierung: Co-Autorschaft

**Transparenz:**
- Autoren-Beiträge dokumentiert (CRediT Taxonomy)
- Diskussion vor Paper-Submission

---

## ❓ FAQ für Contributors

**Q: Kann ich für meine Masterarbeit beitragen?**  
A: Ja! Kontaktiere uns für Themenvorschläge.

**Q: Muss ich Neurowissenschaft verstehen?**  
A: Nein, ML/Software-Engineering-Skills sind auch wertvoll.

**Q: Was wenn mein Experiment negative Ergebnisse hat?**  
A: Perfekt! Negative Ergebnisse sind wissenschaftlich wertvoll.

**Q: Kann ich proprietäre Datensätze nutzen?**  
A: Nur wenn Ergebnisse ohne Daten reproduzierbar sind.

**Q: Wer entscheidet über Merge?**  
A: Projekt-Maintainer (aktuell: Thomas Heisig)

**Q: Kann ich Bezahlung für Beiträge erhalten?**  
A: Aktuell nein, da Forschungsprojekt. Bei Grants: möglich.

---

## 📞 Kontakt

**Projekt-Maintainer:**
- Name: Thomas Heisig
- E-Mail: t_heisig@gmx.de
- Location: Ganderkesee, Germany

**Kommunikation:**
- GitHub Issues (bevorzugt)
- GitHub Discussions (für allgemeine Fragen)
- E-Mail (für vertrauliche/institutionelle Anfragen)

**Response-Zeit:**
- Issues: ~1 Woche
- PRs: ~2 Wochen
- Wissenschaftliche Kooperationen: individuell

---

## 📚 Weitere Ressourcen

- [Code of Conduct](../../CODE_OF_CONDUCT.md)
- [Developer Guide](../developer-guide/README.md)
- [Coding Standards](../developer-guide/coding-standards.md)
- [Scientific Hypotheses](../SCIENTIFIC_HYPOTHESES.md)
- [Research Scope](research-scope.md)
- [Status Overview](status.md)

---

*Letzte Aktualisierung: Dezember 2025*  
*Version: 2.0 (Research-Oriented)*  
*Basierend auf CONTRIBUTING.md, angepasst für wissenschaftlichen Kontext*
