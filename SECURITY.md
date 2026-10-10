# Security Policy

## Supported versions

This is a research project. Security fixes are applied to the latest commit on
`main`. Older commits, tags, Docker images, and independently deployed instances
are not maintained as separate security release branches.

Deployments must be updated and rebuilt by their operators after a fix is released.

## Reporting a vulnerability

Please report vulnerabilities privately. Do not include vulnerability details,
credentials, private datasets, or exploit code in public GitHub issues or pull
requests.

- Email the maintainer at [kainguyen615@gmail.com](mailto:kainguyen615@gmail.com).
  Use the subject `Security report: UAV-link-quality-routing-support`.
- If GitHub private vulnerability reporting is enabled, use
  [Report a vulnerability](https://github.com/kaing615/UAV-link-quality-routing-support/security/advisories/new).

Include:

1. The affected component and commit SHA or image tag.
2. A description of the issue and its potential impact.
3. Minimal reproduction steps using synthetic data, without real credentials.
4. Relevant configuration, with sensitive values removed.
5. A suggested fix, if available, and your preferred attribution.

Reports concerning the API, dashboard, model loading, preprocessing, dependencies,
Docker images, and CI/CD configuration are welcome. For vulnerabilities in a
third-party dependency itself, also follow that dependency's reporting policy.

## Handling reports

The maintainer will assess the report, coordinate a fix where feasible, and discuss
public disclosure with the reporter. This project has no guaranteed response or
resolution deadline and no paid bug bounty program.

Only test systems you own or have permission to test. Use a local environment and
synthetic data where possible; avoid disrupting running experiments or services.

## Deployment considerations

- Keep credentials out of Git, container images, logs, and vulnerability reports.
- Use trusted model artifacts and update dependencies when security fixes are available.
- Protect access to the API, dashboard, and administration tools when exposing a deployment.
- A security policy does not certify the application or an individual deployment as secure.
