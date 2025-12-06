# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

The 4D Neural Cognition team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Preferred**: Open a [GitHub Security Advisory](https://github.com/Thomas-Heisig/4D-Neural-Cognition/security/advisories/new)
2. **Alternative**: Email the maintainers (see CONTRIBUTING.md for contact information)

### What to Include

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: Within 48 hours of report submission
- **Initial Assessment**: Within 7 days
- **Fix Development**: Depends on severity and complexity
- **Public Disclosure**: After fix is available and users have had time to update

### Security Best Practices

When using 4D Neural Cognition, please follow these security best practices:

#### For Deployment

1. **Never expose the web interface directly to the internet** without proper authentication
2. **Use environment variables** for sensitive configuration (not hardcoded values)
3. **Validate all user inputs** before processing
4. **Run with minimum required privileges**
5. **Keep dependencies up to date** regularly

#### For Development

1. **Never commit secrets** (API keys, passwords, etc.) to the repository
2. **Use virtual environments** to isolate dependencies
3. **Review dependencies** for known vulnerabilities regularly
4. **Sanitize file paths** when loading/saving files
5. **Implement rate limiting** for web API endpoints

### Known Security Considerations

The following security considerations are documented in [ISSUES.md](ISSUES.md):

1. **Flask Secret Key**: Should be configured via environment variable
2. **Input Validation**: Web API endpoints need enhanced validation
3. **File Path Injection**: User-provided paths should be validated
4. **No Rate Limiting**: Consider adding for production deployments
5. **Pickle Usage**: Avoided in storage modules for security

### Security Updates

Security updates will be:

1. Released as patch versions (e.g., 1.0.1, 1.0.2)
2. Announced in [CHANGELOG.md](CHANGELOG.md)
3. Tagged with `security` label in GitHub releases
4. Documented with CVE identifiers when applicable

### Scope

This security policy applies to:

- Core simulation engine (`src/`)
- Web application (`app.py`, `templates/`, `static/`)
- Storage and persistence systems
- API endpoints

Out of scope:

- Third-party dependencies (report to respective projects)
- User-created configurations or models
- Intentional misuse of the simulation framework

### Recognition

We appreciate security researchers who help improve the project. With your permission, we will:

- Acknowledge you in the CHANGELOG
- Credit you in the security advisory
- List you in our security hall of fame (if we create one)

Thank you for helping keep 4D Neural Cognition and our users safe!

---

*Last Updated: December 2025*  
*Version: 1.0*
