# Deployment Guide

This document describes how code moves through the pipeline and how
staging / production environments are built.

## Pipeline overview

```
commit → build → test → staging → prod
```

## Branching and merge path

```
feature/*  ──PR──▶  dev  ──PR──▶  staging  ──PR──▶  main
```

- `feature/*` — individual work, opened against `dev`
- `dev` — integration branch
- `staging` — pre-production validation
- `main` — production

Protected branches (`dev`, `staging`, `main`) require CI to pass before a
pull request can be merged.

## Which branch triggers which pipeline

| Trigger | Pipeline | What happens |
|---------|----------|--------------|
| PR to `dev` / `staging` / `main` | CI (`lint-test`, `docker-build`) | Run tests, lint/format/static analysis, build Docker image |
| Merge/push to `staging` | CD (`deploy-staging`) | Create `staging.env` from staging secrets, build staging image |
| Merge/push to `main` | CD (`deploy-prod`) | Create `production.env` from production secrets, build production image |

> Note: no release or runtime deployment process is required by the course.
> The CD stages only build the images.

## CI requirements (`.github/workflows/ci.yml`)

On every pull request to a protected branch:

1. Run the full test suite (`pytest`).
2. Run linting / formatting / static analysis (`ruff`, `mypy`, `bandit`, ...).
3. Build the Docker image and ensure the build succeeds.
4. Block merges when CI fails.

## Required GitHub secrets

Configure under **Settings → Secrets and variables → Actions**.

| Secret | Environment | Purpose |
|--------|-------------|---------|
| `STAGING_ENV` | staging | Content of the staging `.env` file |
| `PROD_ENV` | production | Content of the production `.env` file |
| `AZURE_CREDENTIALS` | both | (optional) DVC Azure remote access |
| `GITHUB_TOKEN` | both | Provided automatically by Actions |

## Docker images

The `Dockerfile` defines two targets:

| Target | Command | Purpose |
|--------|---------|---------|
| `training` | `docker build --target training -t sentiment-analyzer:train .` | Train / fine-tune the model |
| `serving` | `docker build --target serving -t sentiment-analyzer:serve .` | Run the FastAPI service |

Run the service locally:

```bash
docker build --target serving -t sentiment-analyzer:serve .
docker run -p 8000:8000 sentiment-analyzer:serve
```

## Environment files

Staging and production `.env` files are generated in CI from GitHub secrets
and are never committed to the repository.
