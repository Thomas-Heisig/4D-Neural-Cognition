# Documentation Standards

This document describes the documentation standards and organization principles used in the 4D Neural Cognition project.

## Standards Compliance

This project's documentation follows **ISO/IEC/IEEE 26512:2018** - Systems and software engineering — Requirements for acquirers and suppliers of information for users.

### Key Principles

1. **User-Centered**: Documentation organized by user role and task
2. **Comprehensive**: Complete coverage of all aspects
3. **Accessible**: Clear language, multiple entry points
4. **Maintainable**: Logical structure, easy to update
5. **International**: Multilingual support (EN primary, DE partial)
6. **Professional**: Security policy, support structure, community guidelines

## Documentation Organization

### Hierarchical Structure

```
Root Level                    Purpose: Project overview and policies
├── README.md                 Main entry point
├── DOCUMENTATION.md          Documentation hub
├── VISION.md                 Project goals
├── TODO.md                   Planned features
├── CHANGELOG.md              Version history
├── ISSUES.md                 Known issues
├── CONTRIBUTING.md           Contribution guide
├── CODE_OF_CONDUCT.md        Community standards
├── SUPPORT.md                Getting help
└── SECURITY.md               Security policy

docs/                         Purpose: Detailed documentation
├── INDEX.md                  Complete navigation
├── ARCHITECTURE.md           Technical design
│
├── user-guide/              Purpose: End-user documentation
│   ├── README.md            User guide index (EN)
│   ├── README_DE.md         Benutzerhandbuch (DE)
│   ├── INSTALLATION.md      Setup instructions
│   ├── FAQ.md               Common questions
│   ├── GLOSSARY.md          Terminology
│   └── TASKS_AND_EVALUATION.md  Benchmarking
│
├── developer-guide/         Purpose: Contributor documentation
│   └── README.md            Developer guide index
│
├── api/                     Purpose: Technical reference
│   └── API.md               Complete API docs
│
└── tutorials/               Purpose: Learning guides
    └── QUICK_START_EVALUATION.md  5-minute start
```

## File Naming Conventions

### Standard Names (Always Uppercase)

- `README.md` - Main documentation file in any directory
- `CHANGELOG.md` - Version history (Keep a Changelog format)
- `CONTRIBUTING.md` - Contribution guidelines
- `CODE_OF_CONDUCT.md` - Community standards
- `LICENSE` - License file (no extension)
- `SECURITY.md` - Security policy
- `SUPPORT.md` - Support resources

### Descriptive Names

- Use `SCREAMING_SNAKE_CASE` for important docs: `FAQ.md`, `GLOSSARY.md`, `VISION.md`
- Use descriptive names: `INSTALLATION.md`, `ARCHITECTURE.md`
- Add language suffix for translations: `README_DE.md`, `FAQ_DE.md`

### Organization

- Group by audience: `user-guide/`, `developer-guide/`
- Group by type: `api/`, `tutorials/`
- Keep flat hierarchy (max 2-3 levels deep)

## Content Standards

### Document Structure

Every documentation file should have:

1. **Title** (H1) - Clear, descriptive
2. **Introduction** - What this document covers
3. **Table of Contents** (for longer docs)
4. **Body** - Well-organized sections
5. **Related Resources** - Cross-references
6. **Metadata** - Last updated, version

### Writing Style

- **Clear and Concise**: Short sentences, active voice
- **User-Focused**: Address reader directly ("you")
- **Consistent Terminology**: Use glossary terms
- **Code Examples**: Practical, runnable examples
- **Visual Aids**: Diagrams, tables, code blocks

### Formatting

- Use **Markdown** for all documentation
- Use proper heading hierarchy (H1 → H2 → H3)
- Use code blocks with language specification: ```python
- Use tables for structured data
- Use lists for sequential or grouped items
- Use blockquotes for important notes: > **Note**: ...
- Use emoji sparingly for navigation: 📚 📖 🔧 ✅

## Cross-Referencing

### Internal Links

Use relative paths:
```markdown
[User Guide](docs/user-guide/README.md)
[FAQ](docs/user-guide/FAQ.md)
[API Reference](docs/api/API.md)
```

### External Links

Use full URLs with descriptive text:
```markdown
[Python Documentation](https://docs.python.org/)
[ISO/IEC/IEEE 26512](https://www.iso.org/standard/43073.html)
```

### Anchor Links

For same-page navigation:
```markdown
[Installation](#installation)
[Getting Started](#getting-started)
```

## Language Support

### Primary Language: English

- All documentation available in English
- English is the language of code, comments, and APIs
- Technical documentation primarily in English

### Secondary Language: German

- Main README has German section
- User guide available in German (README_DE.md)
- Partial translation of key user-facing documents

### Adding New Languages

To add a new language:

1. Create `README_XX.md` in relevant directories (XX = language code)
2. Translate user-facing documentation first
3. Keep technical documentation in English
4. Update main README with language links
5. Document translation status

## Documentation Types

### User Documentation

**Audience**: End users, researchers, students

**Content**:
- Installation instructions
- Quick start guides
- Usage examples
- FAQ
- Troubleshooting
- Glossary

**Location**: `docs/user-guide/`

### Developer Documentation

**Audience**: Contributors, maintainers

**Content**:
- Development setup
- Architecture overview
- Coding standards
- Testing guidelines
- Contribution workflow
- API design

**Location**: `docs/developer-guide/`

### API Documentation

**Audience**: Developers, power users

**Content**:
- Function signatures
- Parameters and return values
- Code examples
- Error handling
- Best practices

**Location**: `docs/api/`

### Tutorials

**Audience**: New users, learners

**Content**:
- Step-by-step guides
- Working examples
- Common workflows
- Best practices

**Location**: `docs/tutorials/`

## Maintenance

### Regular Updates

- Review documentation with each release
- Update version numbers and dates
- Check and fix broken links
- Update examples for API changes
- Add new FAQ entries from user questions

### Version Control

- All documentation in Git
- Meaningful commit messages
- Document major changes in CHANGELOG.md
- Tag documentation versions with releases

### Quality Checks

Before committing documentation:

- [ ] Spell check
- [ ] Grammar check
- [ ] Link validation
- [ ] Code example testing
- [ ] Cross-reference verification
- [ ] Formatting consistency

## Best Practices

### For Writers

1. **Know Your Audience**: Write for specific user roles
2. **Use Examples**: Show, don't just tell
3. **Be Complete**: Cover all common scenarios
4. **Be Accurate**: Test all instructions and examples
5. **Be Consistent**: Use same terms and style throughout
6. **Be Accessible**: Plain language, clear structure

### For Maintainers

1. **Keep It Current**: Update with code changes
2. **Review Pull Requests**: Check doc updates in PRs
3. **Encourage Contributions**: Welcome doc improvements
4. **Monitor Issues**: Add FAQ entries from common questions
5. **Track Metrics**: Monitor doc usage and feedback
6. **Iterate**: Continuously improve based on feedback

### For Contributors

1. **Update Docs with Code**: Doc updates in same PR as code
2. **Follow Standards**: Use this guide
3. **Ask Questions**: Clarify before documenting
4. **Test Examples**: Ensure code samples work
5. **Get Review**: Have documentation reviewed

## Tools and Resources

### Markdown

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Markdown Cheatsheet](https://www.markdownguide.org/cheat-sheet/)

### Documentation Standards

- [ISO/IEC/IEEE 26512:2018](https://www.iso.org/standard/43073.html)
- [Write the Docs](https://www.writethedocs.org/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)

### Version Control

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Writing Tools

- Spell checkers (built into most editors)
- Grammar checkers (Grammarly, LanguageTool)
- Link checkers (markdown-link-check)
- Linters (markdownlint)

## Metrics

Current documentation metrics (December 2025):

- **Total Files**: 23 markdown files
- **Total Lines**: ~8,000 lines
- **Total Words**: ~50,000 words
- **Languages**: 2 (English, German partial)
- **User Guide Pages**: 6
- **Developer Guide Pages**: 1 (with more planned)
- **API Documentation**: Complete
- **Tutorials**: 1 (more planned)

## Future Improvements

Planned documentation enhancements:

- [ ] Complete German translation
- [ ] Add video tutorials
- [ ] Interactive examples (Jupyter notebooks)
- [ ] Auto-generated API docs from docstrings
- [ ] Documentation testing (link checking, example validation)
- [ ] User feedback system
- [ ] Documentation analytics
- [ ] Additional language support

## Feedback

Help us improve documentation:

- Report issues: [GitHub Issues](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues)
- Suggest improvements: [GitHub Discussions](https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions)
- Contribute: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*Last Updated: December 2025*  
*Standards Version: 1.0*  
*Based on: ISO/IEC/IEEE 26512:2018*
