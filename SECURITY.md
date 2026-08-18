# Security Policy

[English](SECURITY.md) | [Čeština](SECURITY.cs.md)

<!-- doc-status: living; verified: 2026-08-18 -->
> **Document status:** Living documentation, verified against the current code and published results on 2026-08-18.

## Supported version

Security fixes are applied to the current `main` branch. This is experimental research software and has no stable compatibility guarantee yet.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue for credentials, arbitrary-code-execution risks, or other sensitive findings.

Model checkpoints are data files from an experimental pipeline. Only load checkpoints obtained from a trusted release and verify the published checksum first. The repository does not publish executable pickle-based model weights as release assets.
