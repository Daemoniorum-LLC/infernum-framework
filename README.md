# Persona Framework — Domain-specific AI chat agents for WrenchML (and friends)

*A lightweight, domain-oriented chat-agent framework for Spring Boot apps.*

Persona lets you define **domain personas** (Technician, Service Writer, Inventory, Sales, Accounting, Marketing, Management) with their own prompt stacks and policies, then plug in AI providers (starting with **AWS Bedrock**, **SageMaker**, **S3**) via clean ports/adapters. An optional **NL→SQL** module adds governed database querying to any persona.



## What’s New

* **PostgreSQL-first schema & seed**: `V001__persona_schema.sql` and `V002__persona_seed.sql` (timestamptz/jsonb, `BIGSERIAL`, `updated_at` triggers).
* **AI Catalog & Routing**: `ai_provider`, `ai_credential`, `ai_integration`, `ai_routing_policy`, and persona bindings in `persona_integration_binding` (with `fallback_integration_ids` CSV).
* **Safety pipeline**: **Supervisor Agent + Guardrails + Deterministic Verifier** enforcing "no action without evidence" for mission-critical personas.
* **Admin UI starter**: optional React UI bundle served by Spring Boot to manage personas, prompts, and integrations.
* **Seeded prompts**: WrenchML diagnostic JSON template, data normalization template, and an assistant system voice for all personas.
* **AWS adapters**: Bedrock (Converse), Titan embeddings, SageMaker predict, S3 blob ops (GET/PUT).
* **Knowledge base management**: `research_source`, `research_citation`, `research_official_domain` tables with URL validation, S3 integration hooks, and Research Assistant bindings.
* **RAG source catalog**: `persona_rag_source` table linking personas to S3/URL/DB/VECTOR retrieval sources.
* **Usage analytics ready**: `ai_usage_log` with persona/provider/integration tracking, cost attribution, and append-only audit trail.

---

## Module Layout

```
persona-framework/
├─ settings.gradle.kts
├─ build.gradle.kts                 # root (conventions + BOMs)
├─ persona-core/                    # domain model + ports (no Spring)
├─ persona-persistence/             # Spring Data JPA adapters (PostgreSQL)
├─ persona-ai-api/                  # AI orchestration ports + DTOs
├─ persona-ai-aws/                  # AWS adapters (Bedrock, SageMaker, S3)
├─ persona-nl2sql/                  # optional natural language → SQL
├─ persona-autoconfigure/           # Spring Boot autoconfiguration
├─ admin-ui/                        # React app (source)
├─ admin-ui-resources/              # built static assets
└─ admin-ui-starter/                # tiny starter to mount the UI routes
```

**Dependency rule**

```
core  <-- persistence
core  <-- ai-api
ai-api <-- ai-aws
core  <-- nl2sql   (optional)
autoconfigure depends on: core, persistence, ai-api (+ ai-aws / nl2sql if present)
admin-ui(-resources/-starter) are optional at runtime
```

No module may introduce `core → persistence` (avoid build cycles).

---

## Requirements

* **Java 21+**
* **Gradle 8.13+**
* **PostgreSQL 13+** (no H2)
* AWS credentials (if you use AWS adapters)

---

## Getting Started

### 1) Add dependencies to your Spring Boot app

```kotlin
dependencies {
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.4.9"))

    implementation("com.lightspeeddms.persona:persona-core:0.1.0")
    implementation("com.lightspeeddms.persona:persona-persistence:0.1.0")
    implementation("com.lightspeeddms.persona:persona-ai-api:0.1.0")
    implementation("com.lightspeeddms.persona:persona-ai-aws:0.1.0")
    implementation("com.lightspeeddms.persona:persona-autoconfigure:0.1.0")

    // Optional
    implementation("com.lightspeeddms.persona:persona-nl2sql:0.1.0")
    implementation("com.lightspeeddms.persona:admin-ui-starter:0.1.0")
}
```

### 2) Database & Migrations (PostgreSQL only)

Place these two files on your classpath (e.g., `src/main/resources/db/migration`):

* `V001__persona_schema.sql` — creates **personas & config** tables and **AI Catalog & Routing** tables with `updated_at` triggers.
* `V002__persona_seed.sql` — idempotent seed: providers, credentials (AWS default), integrations (Bedrock/Titan/S3/SageMaker), prompts & guardrails, bindings, routing policy.

Spring config:

```yaml
spring:
  datasource:
    url: jdbc:postgresql://172.17.0.1:5432/persona
    username: postgres
    password: 
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        format_sql: true
        jdbc.time_zone: UTC
  flyway:
    enabled: true
    locations: classpath:db/migration
```

> **Testing**: use **Testcontainers** (`jdbc:tc:postgresql:16:///persona`) for integration tests.

### 3) Configure AWS adapters (example)

```yaml
persona:
  ai:
    default-provider: aws
    aws:
      region: us-west-2
      chat:
        model-id: anthropic.claude-3-haiku-20240307-v1:0
        max-tokens: 2000
        temperature: 0.1
      embed:
        model-id: amazon.titan-embed-text-v2:0
      sagemaker:
        endpoint: estimate-predictor
      s3:
        bucket: persona-assets-dev
```

Env overrides:

| Property                            | Env Var                             |
| ----------------------------------- | ----------------------------------- |
| `persona.ai.aws.region`             | `PERSONA_AI_AWS_REGION`             |
| `persona.ai.aws.chat.model-id`      | `PERSONA_AI_AWS_CHAT_MODEL_ID`      |
| `persona.ai.aws.embed.model-id`     | `PERSONA_AI_AWS_EMBED_MODEL_ID`     |
| `persona.ai.aws.sagemaker.endpoint` | `PERSONA_AI_AWS_SAGEMAKER_ENDPOINT` |
| `persona.ai.aws.s3.bucket`          | `PERSONA_AI_AWS_S3_BUCKET`          |

Standard AWS envs (`AWS_REGION`, `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) are respected.

---

## Running the Application

### Local Development (bootRun)

```bash
./gradlew :persona-demo:bootRun
```

### Docker Deployment (Jib)

The project uses [Jib](https://github.com/GoogleContainerTools/jib) for containerization without requiring a Dockerfile.

**1. Build Docker image:**

```bash
./gradlew :persona-demo:jibDockerBuild
```

This creates `persona-demo:0.1.0` in your local Docker daemon.

**2. Create environment file** (`persona-demo/.env`):

```env
SPRING_DATASOURCE_URL=jdbc:postgresql://172.17.0.1:5432/persona
SPRING_DATASOURCE_USERNAME=postgres
SPRING_DATASOURCE_PASSWORD=yourpassword
AWS_REGION=us-west-2
AWS_PROFILE=dev
```

**3. Run container:**

```bash
cd persona-demo
docker run -d \
  --name persona-demo \
  --network host \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  persona-demo:0.1.0
```

**4. Check logs:**

```bash
docker logs -f persona-demo
```

**5. Test endpoints:**

```bash
# Health check
curl http://localhost:8080/actuator/health

# List personas
curl http://localhost:8080/api/personas | jq

# Chat with Software Engineer persona
curl -X POST http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}' | jq

# View usage logs
curl http://localhost:8080/api/usage/logs | jq

# Export usage to CSV
curl http://localhost:8080/api/usage/export?format=csv

# Chat with RAG context retrieval enabled
curl -X POST 'http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER?useRag=true' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What are Spring Boot configuration best practices?"}],"stream":false}' | jq

# List RAG sources for a persona
curl http://localhost:8080/api/knowledge/personas/SR_SOFTWARE_ENGINEER/rag-sources | jq
```

**Notes:**
- `--network host` allows container to access PostgreSQL on host machine
- `-v ~/.aws:/root/.aws:ro` mounts AWS credentials (read-only)
- Ensure AWS SSO session is active: `aws sso login --profile dev`
- Container uses Java 21 (Amazon Corretto Alpine)
- Mount stub-s3 directory for local RAG testing: `-v /path/to/build/stub-s3:/app/build/stub-s3:ro`

---

## RAG (Retrieval Augmented Generation)

The framework supports RAG context retrieval from multiple source types to enhance chat responses with relevant information.

### RAG Source Types

- **S3**: Fetch documents from S3 buckets (or local filesystem in stub mode)
- **URL**: Retrieve content from HTTP/HTTPS URLs
- **DB**: Execute database queries for structured data
- **VECTOR**: Perform vector similarity search (requires vector database)

### Configuration

Enable RAG retrieval on a per-request basis using the `useRag` query parameter:

```bash
# Without RAG (default)
POST /api/public/chat/{personaCode}

# With RAG enabled
POST /api/public/chat/{personaCode}?useRag=true
```

### Managing RAG Sources

RAG sources are stored in the `persona_rag_source` table and bound to specific personas:

```bash
# List RAG sources for a persona
GET /api/knowledge/personas/{personaCode}/rag-sources

# Response:
# [
#   {
#     "id": "uuid",
#     "personaCode": "SR_SOFTWARE_ENGINEER",
#     "sourceType": "S3",
#     "uri": "s3://bucket/key",
#     "metadata": {"description": "..."},
#     "createdAt": "2025-10-18T..."
#   }
# ]
```

### Development Mode (Stub Implementations)

For local development without AWS dependencies, the framework includes stub implementations:

```yaml
persona:
  ai:
    aws:
      s3:
        stub:
          enabled: true
          path: ./build/stub-s3  # Local filesystem path
    rag:
      stub:
        enabled: true  # Use stub fetchers for URL/DB/VECTOR
```

**Stub fetchers:**
- `StubS3ContentFetcher`: Reads from local `build/stub-s3/{bucket}/{key}` directory
- `StubUrlContentFetcher`: Returns placeholder HTTP content
- `StubDatabaseQueryExecutor`: Returns placeholder SQL results
- `StubVectorSearchExecutor`: Returns placeholder search results

### Example RAG Workflow

1. **Create RAG sources** (via migration or API):
```sql
INSERT INTO persona_rag_source (id, persona_id, source_type, uri, metadata_json)
VALUES (
  gen_random_uuid(),
  (SELECT id FROM persona WHERE code = 'SR_SOFTWARE_ENGINEER'),
  'S3',
  's3://docs/spring-boot-guide.md',
  '{"description": "Spring Boot best practices"}'::jsonb
);
```

2. **Place content in stub-s3** (development):
```bash
mkdir -p build/stub-s3/docs
echo "# Spring Boot Best Practices..." > build/stub-s3/docs/spring-boot-guide.md
```

3. **Chat with RAG enabled**:
```bash
curl -X POST 'http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER?useRag=true' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What are Spring Boot best practices?"}],"stream":false}'
```

The system will:
- Retrieve RAG sources for the persona from the database
- Fetch content from each source (S3, URL, DB, VECTOR)
- Append retrieved context to the system message
- Send enriched prompt to the AI model
- Return contextually enhanced response

### Token Budget Management

RAG retrieval respects token limits to avoid exceeding model context windows:

- Default: 2000 tokens for RAG context
- Estimation: ~0.75 tokens per word
- Retrieval stops when budget is reached
- Sources are processed in order until limit

---

## Safety Pipeline (New)

**Supervisor Agent** sits between persona outputs and users, enforcing:

* **Deterministic verification** against manuals/spec DBs (no “trust the LLM”).
* **Guardrails** (e.g., forbid disabling safety systems; enforce spec ranges).
* **Evidence requirement** (every actionable claim must include provenance).
* **Confidence thresholds** (e.g., ≥0.95 for technician instructions).
* **Auditable outcomes** (append-only logs + usage metrics).

Minimal flow:

1. Persona builds prompt → model returns **structured claims** (JSON).
2. Verifier confirms each claim’s evidence against authoritative sources.
3. Guardrails evaluate text for safety/compliance.
4. Supervisor approves or escalates; all decisions are logged.

---

## AI Catalog & Routing (New)

Centralized, DB-driven configuration:

* **Providers** (`ai_provider`): AWS / OpenAI / Azure / GCP / …
* **Credentials** (`ai_credential`): references to secret managers.
* **Integrations** (`ai_integration`): operation + model/endpoint/region.
* **Bindings** (`persona_integration_binding`): persona→primary integration (+ CSV fallbacks).
* **Policies** (`ai_routing_policy`): weighted/priority routing by `selector_json`.

Example binding (seeded):

* All personas’ `CHAT` → **Bedrock Haiku** (@ `us-west-2`).
* Technician `PREDICT` → **SageMaker** endpoint (`estimate-predictor`) with optional fallbacks.

---

## Admin UI (Optional)

Include `admin-ui-starter` to serve the built React bundle:

```kotlin
implementation("com.lightspeeddms.persona:admin-ui-starter:0.1.0")
```

Expose it (dev example):

```kotlin
@Bean
@Order(100)
fun personaAdminUiChain(http: HttpSecurity): SecurityFilterChain =
    http.securityMatcher("/persona-admin/**")
        .authorizeHttpRequests { it.anyRequest().permitAll() }
        .csrf { it.disable() }
        .httpBasic { it.disable() }
        .formLogin { it.disable() }
        .build()
```

Default path: `/persona-admin`.

---

## REST API Endpoints

### Chat
- `POST /api/public/chat/{personaCode}` – Chat with a persona

### Embeddings
- `POST /api/public/embed` – Generate embeddings for texts

### Natural Language to SQL
- `POST /api/public/nl2sql/{personaCode}` – Generate SQL from natural language

### Usage Analytics (NEW)
- `GET /api/usage/logs` – List usage logs with filters (persona, provider, operation, date range)
- `GET /api/usage/summary` – Aggregated cost/token statistics
- `GET /api/usage/export` – Export usage logs as CSV

### Knowledge Base (NEW)
- `GET /api/knowledge/sources` – List research sources
- `POST /api/knowledge/sources` – Store a new research source
- `GET /api/knowledge/citations/{sourceId}` – Get citations for a source
- `POST /api/knowledge/sources/{id}/attach-s3` – Attach S3 object to source
- `GET /api/personas/{code}/rag-sources` – List RAG sources bound to persona

### Admin
- `GET /api/personas` – List all personas
- OpenAPI docs available at `/swagger-ui.html`

---

## Using Personas in Code

### Chat with a Persona

```kotlin
@Service
class SupportAgent(
  private val personas: PersonaRepositoryPort,
  private val prompts: PersonaPromptRepositoryPort,
  private val chat: ChatClient
) {
  fun answer(code: String, userText: String): String {
    val persona = personas.findByCode(code) ?: error("persona not found: $code")
    val systemStack = prompts.findByPersonaId(persona.id)
      .filter { it.kind == "SYSTEM" }
      .sortedBy { it.version }
      .joinToString("\n") { it.content }
    val messages = listOf(ChatMessage.system(systemStack), ChatMessage.user(userText))
    return chat.complete(messages).text
  }
}
```

### NL→SQL (Optional)

```kotlin
class ReportsService(private val nl2sql: Nl2SqlClient) {
  fun ask(reportIntent: String): String =
    nl2sql.generate(
      personaCode = "ACCOUNTING",
      instruction = reportIntent,
      constraints = mapOf("schemas" to listOf("public","acct"), "max_rows" to 1000),
      dryRun = true
    )
}
```

---

## Build & Test

```bash
./gradlew clean build
```

Testcontainers example (`src/test/resources/application-test.yml`):

```yaml
spring:
  datasource:
    url: jdbc:tc:postgresql:16:///persona
    driver-class-name: org.testcontainers.jdbc.ContainerDatabaseDriver
  jpa:
    hibernate.ddl-auto: validate
  flyway.enabled: true
```

---

## Troubleshooting

* **Build cycles**: ensure `core` does not depend on `persistence`.
* **Spring Data types not found**: add Boot BOM and `spring-boot-starter-data-jpa` to `persona-persistence`.
* **Kotlin JPA plugin**: apply `kotlin("plugin.jpa")` with a version in module `plugins`.
* **RepositoriesMode error**: define repos in `settings.gradle.kts`.
* **JSONB errors**: you’re not on Postgres—switch to Postgres (no H2).

---

## Roadmap

### High Priority
* **Usage analytics endpoints** – expose ai_usage_log via REST API with cost dashboards
* **S3 adapter completion** – implement presigned URLs and research_source attachment workflows
* **RAG retrieval service** – build vector search + knowledge base query orchestration
* **Knowledge base UI** – admin panel for research_source, research_citation, persona_rag_source management

### Medium Priority
* Streaming interfaces (server-sent tokens) across all adapters
* SageMaker adapter completion (endpoint invocation + batch transform)
* Tool/Function calling DSL + validation schema
* Prompt bundle versioning via S3 manifests & A/B testing

### Future
* Multi-tenant partitioning (workspace/org)
* OpenTelemetry tracing + distributed cost attribution
* Expanded guardrails (regex + semantic + policy graph)
* Deeper WrenchML integration using Supervisor + evidence verification for technician workflows

---

## Contributing

1. Fork and create a feature branch.
2. Keep modules acyclic (`./gradlew :persona-core:dependencies`).
3. Add tests (prefer Testcontainers).
4. Open a PR with a clear description.

---

## License

© 2025 LightspeedDMS. All rights reserved. 

---

## Appendix: ASCII Package/Dependency Diagram

```
[ persona-core ]  <-- domain, ports
      ^
      |
[ persona-persistence ] -- Spring Data JPA (PostgreSQL)
      ^
      |
[ persona-autoconfigure ] -- Boot configs/wiring
      ^
      |
[ persona-ai-api ] <---[ persona-ai-aws ] (Bedrock, SageMaker, S3)

[ persona-nl2sql ] (optional) --> plugs into ai-api + core ports

[ admin-ui ] (React) -> [ admin-ui-resources ] -> [ admin-ui-starter ] -> Spring Boot static route
```



## WSL Smoke Tests

To run backend smoke tests from Windows using WSL (no PowerShell tooling), use:

- Ensure the backend is running (e.g., `./gradlew :persona-demo:bootRun` inside WSL or another terminal).
- From a Windows prompt, invoke WSL and run the Linux script:

```
wsl bash -lc "./scripts/smoke.sh"
```

Environment variables supported:
- PERSONA_SERVER_URL: base server URL (default http://localhost:8989). A trailing `/api` will be normalized away.
- PERSONA_API_TOKEN: optional bearer token to include in requests.
- PERSONA_CODE: persona code for chat/NL2SQL tests (default SR_SOFTWARE_ENGINEER).

The script requires `curl` and `jq` inside WSL. It verifies:
- GET /api/personas (200, JSON array)
- POST /api/public/chat/{personaCode} (returns text)
- POST /api/nl2sql/{personaCode} (returns expected fields)
- POST /api/embed (returns embeddings)
