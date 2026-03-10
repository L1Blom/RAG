# X Upload Media Ingestion Design (No Video/Audio Indexing Yet)

## Goal
Enable `/prompt/<project>/uploadx` to download and persist X post assets under `data/<ID>/` while keeping current document handlers responsible for all non-X formats.

This design adds:
- Per-post storage structure: `data/<tweet_id>/`.
- Dedicated media folders for X posts:
  - `data/<tweet_id>/images/`
  - `data/<tweet_id>/videos/`
  - `data/<tweet_id>/audio/` (optional folder creation only, no processing/indexing in this phase)
- Controlled migration from existing `x.json` workflow to local snapshots.
- `/reload` rebuild mode that clears vector DB first, then re-indexes from local files.

This design does not add:
- Video embedding/transcription.
- Audio embedding/transcription.
- Multimodal indexing of binary media.

## Current State (Observed)
- `/uploadx` stores URL in `data_dir/x.json` and calls `vector_store_svc.load_x_urls([url])`.
  - File: `rag/api/routes.py`
- X loader fetches tweet text/metadata from X API and writes text chunks to vector store.
  - File: `rag/services/document_loader/x_loader.py`
- `/reload` triggers `initialize_chain(True)` which reloads all registered loaders.
  - File: `rag/api/routes.py`
- Loader registry runs the X loader via `x.json` and all other file handlers via extension globs.
  - File: `rag/services/document_loader/registry.py`

## Requirements Coverage
- X posts are uploaded via `/uploadx` and `/uploadx/batch`.
- Media from X posts is stored in per-post folders under `data/<ID>`.
- Other formats remain handled by existing document handlers (unchanged).
- `/reload` rebuild clears vector DB first and then indexes X text context from local artifacts only.
- `/reload` must not refetch from X API.
- Video/audio are downloaded and persisted only; no indexing/transformation now.

## Target Data Layout
For tweet id `1234567890`:

```text
data/
  1234567890/
    post.json
    post.txt
    images/
      img_01.jpg
      img_02.png
    videos/
      vid_01.mp4
    audio/
      audio_01.m4a
```

### File Semantics
- `post.json`: canonical metadata snapshot for the X post and downloaded assets.
- `post.txt`: normalized textual representation used for embedding/retrieval.
- `images/`, `videos/`, `audio/`: raw files for future multimodal work.

## Proposed Code Changes

### 1) Extend X loader for snapshot + media download
File: `rag/services/document_loader/x_loader.py`

Add methods:
- `_build_post_dir(self, data_dir: str, tweet_id: str) -> str`
- `_ensure_media_dirs(self, post_dir: str) -> dict`
- `_fetch_tweet_with_media(self, tweet_id: str, max_retries: int = 3) -> Optional[Dict[str, Any]]`
- `_extract_media_urls(self, tweet_data: Dict[str, Any]) -> Dict[str, list]`
- `_download_asset(self, url: str, target_path: str, timeout: int = 30) -> bool`
- `_persist_post_snapshot(self, post_dir: str, post_json: Dict[str, Any], post_text: str) -> None`
- `_tweet_to_text(self, tweet_data: Dict[str, Any], post_url: str) -> str`
- `_load_post_snapshot(self, post_json_path: str) -> Optional[Dict[str, Any]]`

Adjust existing methods:
- `load(...)`:
  - For each URL:
    - fetch tweet + media metadata.
    - persist `post.json` and `post.txt` in `data/<tweet_id>/`.
    - download media into `images/`, `videos/`, `audio/`.
    - create indexable `Document` from `post.txt` only.
  - Keep return value as number of added text splits.
- `_fetch_tweet(...)` can remain and call `_fetch_tweet_with_media(...)` or be replaced.
- `_tweet_to_document(...)` should include metadata pointers to local snapshot paths.

X API request fields (minimum):
- `tweet.fields`: `created_at,author_id,public_metrics,text,source,attachments`
- `expansions`: `author_id,attachments.media_keys`
- `media.fields`: `type,url,preview_image_url,variants,duration_ms`

Notes:
- For videos, choose best available variant URL for storage.
- For audio, if represented as variants or separate media, download to `audio/`.
- If media download fails, still index text (`post.txt`) and record per-asset error in `post.json`.

### 2) Pass `data_dir` through X URL loading path
File: `rag/services/vector_store.py`

Change `load_x_urls(self, x_urls: List[str])` config payload to include:
- `data_dir`: `self.config.data_dir`
- optional controls:
  - `download_media`: `True`
  - `index_text_only`: `True`

Reason:
- X loader needs `data_dir` to create `data/<tweet_id>/...` snapshots when called from `/uploadx`.

### 3) Keep route contract, improve status detail
File: `rag/api/routes.py`

No route shape changes required.

Optional response enhancement:
- `/uploadx` success message includes tweet id and media counts.
- `/uploadx/batch` response includes per-url summary:
  - `tweet_id`
  - `text_chunks`
  - `images_downloaded`
  - `videos_downloaded`
  - `audio_downloaded`

Do not remove `x.json` updates yet.

### 4) Enable strict local `/reload` for X posts (no refetch)
File: `rag/services/document_loader/x_loader.py`

Behavior change for `load(...)` when `x_urls` is absent:
- Source for reload must be local `data/*/post.json` snapshots and `post.txt` only.
- Do not call X API during reload.
- `x.json` may be kept for upload tracking, but is not a reload source.

Rationale:
- `/reload` is a deterministic local rebuild and should not depend on external API availability.
- Keeps existing non-X loaders untouched.

### 5) Add vector DB clear-and-rebuild flow
Files:
- `rag/services/vector_store.py`
- `rag/api/routes.py`

Add service methods:
- `clear_vectorstore(self) -> None`
- `rebuild_documents(self, file_type: str = 'all') -> int`

Expected behavior:
- `clear_vectorstore()` removes current collection contents (or drops/recreates collection) and resets in-memory vectorstore handle.
- `rebuild_documents()` calls clear first, then calls existing reload pipeline.

Route behavior:
- `/reload` should call rebuild semantics (clear + re-index), not additive reload.
- Keep endpoint URL unchanged to avoid API breakage.

### 6) Registry remains unchanged for non-X handlers
File: `rag/services/document_loader/registry.py`

No functional changes required for pdf/txt/html/docx/pptx/xlsx.

Optional enhancement:
- log that X reload now reads local snapshots when present.

## Snapshot Schema (`post.json`)

```json
{
  "tweet_id": "1234567890",
  "post_url": "https://x.com/user/status/1234567890",
  "author": {
    "id": "1111",
    "username": "user",
    "name": "User Name"
  },
  "created_at": "2026-03-10T12:00:00.000Z",
  "text": "Post text...",
  "metrics": {
    "like_count": 10,
    "retweet_count": 2,
    "reply_count": 1,
    "quote_count": 0
  },
  "assets": {
    "images": [
      {
        "url": "https://...",
        "local_path": "data/1234567890/images/img_01.jpg",
        "status": "downloaded"
      }
    ],
    "videos": [
      {
        "url": "https://...",
        "local_path": "data/1234567890/videos/vid_01.mp4",
        "status": "downloaded"
      }
    ],
    "audio": [
      {
        "url": "https://...",
        "local_path": "data/1234567890/audio/audio_01.m4a",
        "status": "downloaded"
      }
    ]
  },
  "indexing": {
    "mode": "text_only",
    "video_indexed": false,
    "audio_indexed": false
  },
  "errors": []
}
```

## Retrieval Metadata Contract
Text chunks generated from `post.txt` should include metadata keys:
- `source = "x"`
- `tweet_id`
- `post_url`
- `author_username`
- `created_at`
- `snapshot_path` (path to `post.json`)
- `text_path` (path to `post.txt`)
- `images_count`
- `videos_count`
- `audio_count`
- `indexing_mode = "text_only"`

## Error Handling
- Missing X token: return 500 from route with clear message (existing pattern).
- Tweet fetch fails (404/private/rate-limit): no text chunks; include detailed error.
- Media download fails: continue; index text if available and record failed assets in `post.json`.
- Batch mode: continue processing remaining URLs; return aggregated status.

## Backward Compatibility
- Continue writing new URLs to `x.json`.
- `load(...)` should support `x_urls` for upload-time ingestion.
- Reload path uses local snapshots only and does not depend on `x.json`.
- Existing `/reload` endpoint contract unchanged.
- Existing non-X document handlers unchanged.

## Security and Stability
- Normalize file names and extensions from content-type + URL path.
- Enforce path safety: never allow traversal outside `data_dir`.
- Set per-download size/time limits.
- Use atomic writes for `post.json` and `post.txt` (tmp file then rename).

## Implementation Phases
1. Phase 1: Snapshot + media download in upload path.
2. Phase 2: Add clear-and-rebuild vector DB flow for `/reload`.
3. Phase 3: Reload X from local snapshots only (no API refetch).
4. Phase 4: Route response enrichment and test hardening.
5. Phase 5 (later): optional video/audio indexing/transcription.

## Acceptance Criteria
1. Upload single X URL creates `data/<tweet_id>/post.json` and `data/<tweet_id>/post.txt`.
2. If post has images, files exist under `data/<tweet_id>/images/`.
3. If post has videos, files exist under `data/<tweet_id>/videos/`.
4. If post has audio, files exist under `data/<tweet_id>/audio/`.
5. Text from X post is indexed and retrievable via existing prompt/search paths.
6. Video/audio are not indexed and not transformed in this phase.
7. `/reload` clears the vector DB before re-indexing.
8. `/reload` re-ingests X text context from local snapshots only (no X API calls).
9. Existing document handlers for non-X files continue to behave unchanged.
10. Batch upload reports per-URL success/failure without aborting the whole batch.
11. Existing `x.json` can remain as upload ledger but is not required for reload.

## Test Plan (Implementation-Ready)
Unit tests:
- `XLoaderStrategy._extract_tweet_id` URL variants.
- media extraction mapping from API payload.
- safe path generation under `data/<tweet_id>/...`.
- partial failure behavior (media fail, text success).

Integration tests:
- `/uploadx` with media post:
  - validates folder/files created.
  - validates text chunks added.
- `/uploadx` with text-only post:
  - no media required; still indexed.
- `/reload`:
  - indexes from local `post.json`/`post.txt` snapshot.
  - makes no outbound X API calls.
  - clears existing vector data before rebuild to avoid duplicates.
- Non-X regression:
  - `reload_documents('all')` continues to load pdf/txt/html/docx/pptx/xlsx as before.

## Rollback Strategy
- Disable media download path via config flag `download_media=false`.
- Keep current additive reload path behind an explicit feature flag if needed.
- No schema migration needed for existing indexed vectors.
