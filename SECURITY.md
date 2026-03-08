# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in ts-autopilot, please report it
responsibly:

1. **Do NOT** open a public GitHub issue for security vulnerabilities.
2. Email the maintainers with a description of the vulnerability.
3. Include steps to reproduce if possible.
4. We will acknowledge receipt within 48 hours.
5. We will provide a fix or mitigation within 7 days for critical issues.

## Security Measures

ts-autopilot implements the following security measures:

- **Input validation**: File paths are resolved and validated against symlink
  and path traversal attacks.
- **SSRF prevention**: Tollama URLs are validated for scheme and resolved IPs
  are checked against private network ranges.
- **XSS prevention**: HTML reports use Jinja2 with autoescape enabled.
- **Dependency scanning**: CI pipeline runs pip-audit for known vulnerabilities.
- **File size limits**: CSV input files are limited to 500 MB.
- **Memory limits**: Loaded DataFrames are checked against configurable memory
  limits (default 2048 MB).
- **Safe YAML loading**: Only `yaml.safe_load()` is used for config files.
