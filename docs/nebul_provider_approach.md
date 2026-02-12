# Phased Approach: Adding the Nebul Provider

> **Status:** Phases 1–7 implemented. Phase 8 (Testing) remains.

## Phase 1 — Configuration & Secrets
**Goal:** Make the system aware of Nebul as a valid provider.

| File | Change |
|------|--------|
| `constants/constants.ini` | Add `NEBUL` to the `LLMS` list, add `[LLMS.NEBUL]` section with `modeltext`, `embedding_model`, and a `base_url` for the Nebul API endpoint |
| `env/config.env` | Add `NEBUL_APIKEY=<your-key>` |
| New file `env/NEBUL.txt` | Store the raw Nebul API key (consistent with other providers) |
| `ragservice.py` | Add `os.environ['NEBUL_APIKEY']` to the Docker secrets loading block |

## Phase 2 — Model Discovery
**Goal:** Allow `get_modelnames()` to list available Nebul models.

| File | Change |
|------|--------|
| `ragservice.py` | Add a `case 'NEBUL':` branch in `get_modelnames()`. Since Nebul follows OpenAI standards, create an `openai.OpenAI(api_key=..., base_url=...)` client and use `client.models.list()` — similar to the OPENAI case, but with the custom `base_url`. You may need to adjust the model-name filtering (the current filter looks for prefixes like `gpt`, `o1`, `o3`) to match Nebul's model naming convention. |

## Phase 3 — Chat Model Instantiation
**Goal:** Wire up LangChain to use Nebul for chat completions.

| File | Change |
|------|--------|
| `ragservice.py` | Add a `case "NEBUL":` branch in `set_chat_model()`. Use `ChatOpenAI` with the extra `openai_api_base` parameter pointing to the Nebul API URL: `ChatOpenAI(api_key=..., base_url=rc.get('LLMS.NEBUL','base_url'), model=rcmodel, temperature=temp)` |

## Phase 4 — Embeddings
**Goal:** Enable Nebul embeddings (or fall back to OpenAI embeddings).

| File | Change |
|------|--------|
| `ragservice.py` | Add a `case "NEBUL":` branch in `embedding_function()`. If Nebul provides its own embeddings endpoint, use `OpenAIEmbeddings(api_key=..., base_url=..., model=...)`. If not, fall back to OpenAI embeddings (like GROQ and OLLAMA do). |

## Phase 5 — Image Processing (optional)
**Goal:** Support image analysis via Nebul if applicable.

| File | Change |
|------|--------|
| `ragservice.py` | Add an `elif globvars['USE_LLM'] == 'NEBUL':` branch in `process_image()`. Since it's OpenAI-compatible, reuse the OPENAI logic but with the Nebul `base_url` and API key. |

## Phase 6 — Config Service & UI Integration
**Goal:** Make Nebul selectable through the config service.

| File | Change |
|------|--------|
| `configservice.py` | No code change needed — it reads `provider` dynamically from the `.ini` file. Just ensure the front-end/UI dropdown (if any) includes `NEBUL` as an option. |
| `config.json` | Will be auto-populated when a new project with `provider: NEBUL` is created. |

## Phase 7 — Docker & Deployment
**Goal:** Make Nebul work in containerised environments.

| File | Change |
|------|--------|
| `docker-compose.yml` | Add the `nebul_apikey` secret reference |
| `Dockerfile` | No change expected (it already copies the full project) |
| `requirements.txt` | No new dependencies needed — Nebul uses the existing `openai` and `langchain-openai` packages since it's OpenAI-compatible |

## Phase 8 — Testing
**Goal:** Validate end-to-end.

| File | Change |
|------|--------|
| `config_unittest.py` | Add a `NEBUL` entry to the `tests` dictionary with expected test prompts/responses |
| `ragservice_unittest.py` | Existing test framework will pick it up — just create a `constants/constants_test_nebul.ini` with `USE_LLM = NEBUL` and run the tests against it |
| Manual test | Start a project with `provider=NEBUL`, issue a prompt, verify the response comes from the Nebul endpoint |

## Summary of Effort

| Phase | Effort | Risk |
|-------|--------|------|
| 1. Configuration & Secrets | Low | Low |
| 2. Model Discovery | Low | Medium — depends on how Nebul's `/models` endpoint responds |
| 3. Chat Model | Low | Low — OpenAI-compatible = `ChatOpenAI` + `base_url` |
| 4. Embeddings | Low | Medium — need to confirm if Nebul offers embeddings |
| 5. Image Processing | Low | Optional — depends on Nebul vision model support |
| 6. Config Service | Minimal | Low |
| 7. Docker | Low | Low |
| 8. Testing | Medium | Low |

The key advantage of Nebul being OpenAI-compatible is that you can reuse `ChatOpenAI` and `OpenAIEmbeddings` from `langchain-openai` — no new dependencies required. The main thing you'll need from Nebul is the **API base URL**, **API key**, and the **model names** they support.
