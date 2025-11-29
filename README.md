# Persona Framework

Autonomous agent platform built with Kotlin/Spring Boot, React, and IntelliJ tooling. The repo hosts the core runtime, REST API, admin UI, IDE plugin, and a collection of verticalized services (art generation and Vulcan microservices).

## What’s in the repo
### Core applications
- **Leviathan** – primary Spring Boot backend with agent orchestration, REST API, tool registry, and integrations (Ollama/AWS/Anthropic/OpenAI). See `leviathan/README.md`.
- **Bael** – React + TypeScript admin UI that talks to Leviathan. See `bael/README.md`.
- **Hydra** – Spring Boot gateway for legacy integrations. See `hydra/README.md`.
- **Paimon** – IntelliJ IDEA plugin for in-editor agent workflows. See `paimon/README.md`.
- **Art Service** – Spring Boot app for digital painting workflows; also embedded in Leviathan. See `art-service/README.md`.
- **Apothecary Service** – herbal chemistry automation workflows. See `apothecary-service/README.md`.
- **Nexus Standalone** – slim Spring Boot packaging that reuses Leviathan while trimming optional modules. See `nexus-standalone/README.md`.

### Shared libraries & integrations
- **Persona Core** – shared DTOs and ports used across the platform. See `persona-core/README.md`.
- **Persona REST** – consolidated REST surface with Workspace/Nexus/agent controllers. See `persona-rest/README.md`.
- **Persona Autoconfigure** – Spring Boot auto-configuration for persistence, agents, and observability. See `persona-autoconfigure/README.md`.
- **Persona API** – Kotlin chat/embedding/NL2SQL client helpers. See `persona-api/README.md`.
- **Persona AWS** – Bedrock, S3, Secrets Manager, and retrieval helpers. See `persona-aws/README.md`.
- **Persona MCP** – MCP command protocol glue with auditing/security. See `persona-mcp/README.md`.
- **Persona Sandbox** – Docker-based sandboxed execution helpers. See `persona-sandbox/README.md`.
- **CNC Common** – shared DTOs, events, and exception handling for CNC services. See `cnc-common/README.md`.

### CNC vertical microservices
- **CNC Integration Hub** – secure CAD upload API for the CNC suite. See `cnc-integration-hub/README.md`.
- **CNC Machine Monitor** – machine telemetry ingestion and query API. See `cnc-machine-monitor/README.md`.
- **CNC Portal Service** – portal user management API. See `cnc-portal-service/README.md`.
- **Remaining CNC services** – customer, quote, job, schedule, shop-floor, inventory, quality, accounting, analytics, and other `cnc-*` apps continue to live in their respective folders; look for in-directory READMEs where provided (for example, `cnc-customer-service/README.md`).

### Documentation & knowledge
- **Grimoire** – prompt/persona/tool catalog with `TOOLS.md`, `personas/`, `prompts/`, and `templates/`. See `grimoire/README.md`.

## Prerequisites
- Java 21 (Gradle wrapper is provided)
- Node.js 18+ (for Bael)
- PostgreSQL 13+ (used by Leviathan and the Spring Boot services)
- Optional: Ollama running at `http://localhost:11434` for local ML, Kafka for CNC event streaming, Redis/Elasticsearch if you enable those integrations

## Quick start (Leviathan + Bael)
1. **Configure environment** (example):
   ```bash
   export SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
   export SPRING_DATASOURCE_USERNAME=postgres
   export SPRING_DATASOURCE_PASSWORD=postgres
   export PERSONA_AI_OLLAMA_ENABLED=true
   export PERSONA_AI_OLLAMA_BASE_URL=http://localhost:11434
   # Optional workspace roots
   export WORKSPACE_BASE_PATH=/home/lilith/development/projects/persona-framework/workspace
   export WORKSPACE_PROJECTS_PATH=/home/lilith/development/projects
   ```
   Additional knobs live in `leviathan/src/main/resources/application.yaml`.
2. **Run Leviathan** (default port 8080):
   ```bash
   ./gradlew :leviathan:bootRun
   ```
3. **Run Bael** (default Vite dev server port 5173):
   ```bash
   cd bael
   npm install
   VITE_API_BASE_URL=http://localhost:8080 npm run dev
   ```

## Running other applications
- Hydra: `./gradlew :hydra:bootRun`
- Paimon (IntelliJ plugin dev): `./gradlew :paimon:runIde`
- Art Service: `./gradlew :art-service:bootRun`
- Nexus Standalone: `./gradlew :nexus-standalone:bootRun`
- Vulcan microservices (one at a time, choose a free port): `./gradlew :cnc-customer-service:bootRun` (repeat for other `cnc-*` services)

## Testing
- Backend/agents/services: `./gradlew test`
- Bael unit tests: `cd bael && npm test`
- Bael e2e: `cd bael && npm run e2e`
- Paimon plugin tests: `./gradlew :paimon:test`

## Docker/Jib
- Build Leviathan image: `./gradlew :leviathan:jibDockerBuild`
- Build Hydra image: `./gradlew :hydra:jibDockerBuild`
  (uses Amazon Corretto 21 base image and respects gradle properties `containerName`, `imageTag`, etc.)

## Documentation
- API usage: `docs/API_USAGE_GUIDE.md`
- Developer index: `docs/developer/INDEX.md`
- Additional roadmaps and reports: `docs/` (numerous session summaries and audits)
- Application-specific guides: see the per-application READMEs referenced above.

## Support
For questions or issues, contact **Lilith Crook** (`lilith@daemoniorum.com`) or open a ticket in this repository.
