# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for suspected security vulnerabilities.

Email the maintainer or repository owner with:

- affected version or commit
- reproduction steps
- expected impact
- any logs or screenshots that do not expose secrets

## Secrets

AIterate should never commit provider keys, Git tokens, tracking tokens, or database credentials.
Use encrypted database-backed secret storage for local/self-hosted installs or a managed secret
provider such as Vault, AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager in production.
