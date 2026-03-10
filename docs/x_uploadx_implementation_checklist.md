# X UploadX Implementation Checklist

Scope: implement approved design in `docs/x_uploadx_media_ingestion_design.md`.
Constraint: video/audio are downloaded and persisted, not indexed/transformed.

## 1) `rag/services/vector_store.py`

### Tasks
- Add `clear_vectorstore(self) -> None`.
- Add `rebuild_documents(self, file_type: str = 'all') -> int`.
- Update `load_x_urls(self, x_urls: List[str])` config payload:
  - include `data_dir`
  - include `download_media=True`
  - include `index_text_only=True`

### Notes
- `clear_vectorstore` should guarantee no stale chunks remain.
- Prefer explicit collection delete/reset semantics over additive behavior.

### Done When
- Calling `rebuild_documents('all')` clears then reloads.
- Calling `load_x_urls` passes `data_dir` through to X loader.

## 2) `rag/api/routes.py`

### Tasks
- Change `/prompt/<project>/reload` to call clear-and-rebuild flow.
  - use new `vector_store_svc.rebuild_documents()` path.
- Keep endpoint contract unchanged (`/reload`, same methods).
- Keep `/uploadx` and `/uploadx/batch` contracts unchanged.
- Optional: enrich upload responses with per-post media counts.

### Notes
- `/reload` must be deterministic local rebuild (no X refetch via reload path).

### Done When
- `/reload` returns success after DB clear + re-index.
- Repeated `/reload` does not duplicate old vectors.

## 3) `rag/services/document_loader/x_loader.py`

### Tasks
- Add snapshot/local storage helpers:
  - `_build_post_dir`
  - `_ensure_media_dirs`
  - `_persist_post_snapshot`
  - `_load_post_snapshot`
- Add X media-aware API fetch path:
  - `_fetch_tweet_with_media`
  - `_extract_media_urls`
  - `_download_asset`
- Add text normalization helper:
  - `_tweet_to_text`
- Update `load(...)` behavior:
  - Upload path (`x_urls` present): fetch from X API, persist `post.json` + `post.txt`, download media, index text only.
  - Reload path (`x_urls` absent): load from local snapshots only (`data/*/post.json` + `post.txt`), no X API calls.
- Ensure metadata on indexed chunks includes:
  - `source`, `tweet_id`, `post_url`, `author_username`, `created_at`
  - `snapshot_path`, `text_path`
  - `images_count`, `videos_count`, `audio_count`
  - `indexing_mode=text_only`

### Notes
- Media download failures should not block text indexing.
- Keep `x.json` support for upload ledger; not used as reload source.

### Done When
- Uploading one X URL creates `data/<tweet_id>/post.json` and `post.txt`.
- Media files are downloaded into `images/`, `videos/`, `audio/` when available.
- Reload indexes from local snapshots without outbound X calls.

## 4) `rag/services/document_loader/registry.py`

### Tasks
- Keep non-X loader behavior unchanged.
- Optional: add log annotation indicating X local snapshot reload mode.

### Done When
- Existing `pdf/txt/html/docx/pptx/xlsx` loading remains unchanged.

## 5) Tests

## 5.1 Unit tests (new file suggestion)
- `tests/unit/test_x_loader_media.py`

### Cases
- tweet ID extraction for common URL variants.
- media URL extraction mapping from API payload.
- safe path creation under `data/<tweet_id>/...`.
- partial failure: media fail, text still indexed.
- reload mode does not call API when loading snapshots.

## 5.2 Integration tests
- `tests/unit/test_uploadx_reload_flow.py` (or existing route tests location)

### Cases
- `/uploadx` creates snapshot + media folders.
- `/uploadx` text-only post still indexed.
- `/reload` clears DB first.
- `/reload` rebuild uses local snapshots only.
- non-X reload behavior regression test.

## 6) Operational Checks

### Commands
- Run tests:
  - `pytest tests/unit -q`
- Run focused tests:
  - `pytest tests/unit/test_x_loader_media.py -q`

### Runtime checks
- Confirm `X_API_KEY`/`TWITTER_BEARER_TOKEN` available for upload path.
- Confirm no X API traffic occurs during `/reload` path.

## 7) Suggested Commit Plan

1. `feat(vector-store): add clear-and-rebuild reload semantics`
2. `feat(x-loader): persist snapshots and download media assets`
3. `feat(reload): rebuild from local X snapshots only`
4. `test(x-uploadx): add media and reload coverage`
5. `docs(x-uploadx): finalize behavior and operator notes`

## 8) Acceptance Gate
- [ ] Upload creates `data/<tweet_id>/post.json` and `post.txt`.
- [ ] Images/videos/audio are downloaded when present.
- [ ] Only text content is indexed.
- [ ] `/reload` clears vector DB before rebuild.
- [ ] `/reload` performs no X API calls.
- [ ] Non-X document handlers unchanged.
