# RAGUI Integration Guide For X-Post Uploader

This guide explains how to adapt `../RAGUI` to the current X-post implementation in this backend.

## What Changed In RAG

X upload now stores per-post snapshots and media under:
- `data/<tweet_id>/post.json`
- `data/<tweet_id>/post.txt`
- `data/<tweet_id>/images/*`
- `data/<tweet_id>/videos/*`
- `data/<tweet_id>/audio/*`

Reload behavior changed:
- `/prompt/<project>/reload` clears and rebuilds vector data from local files.
- X reload uses local snapshots (`post.json` + `post.txt`), no required refetch from X.

Upload response enhancement:
- `/prompt/<project>/uploadx` returns text like:
  - `X post stored successfully (3 chunks) tweet_id=123 images=1 videos=2 audio=0`

## API Notes Relevant To UI

1. `GET /prompt/<project>/file?file=x.json`
- Returns the X URL list (JSON array) when present.

2. `GET /prompt/<project>/file?file=<tweet_id>/post.json`
- Returns per-post metadata including `assets` and indexing info.

3. `GET /prompt/<project>/file?file=<tweet_id>/post.txt`
- Returns normalized text that was indexed.

4. `GET /prompt/<project>/file?file=<tweet_id>/images/<name>`
- Serves image file bytes.
- Same pattern for `videos` and `audio`.

5. `GET|POST /prompt/<project>/uploadx?url=...`
- Text response (not JSON). Parse optional summary fields from message.

6. `POST /prompt/<project>/uploadx/batch`
- Text response summary. On single-URL failure returns HTTP 500 with error text.

Important limitation:
- `/prompt/<project>/context?mode=url` reads `urls.json`, not `x.json`.
- For X lists, use `/file?file=x.json`.

## RAGUI Changes (File-By-File)

## 1) `client/src/components/Upload.jsx`

### Change existing X-post list fetch
Replace the current `context?mode=url&file=x.json` call with:
- `GET /prompt/<project>/file?file=x.json`

Reason:
- Current backend `mode=url` always reads `urls.json`.

### Improve X upload result handling
After `uploadx` success:
- Keep existing text display.
- Parse optional fields with regex for better UX:
  - `tweet_id=(\d+)`
  - `images=(\d+)`
  - `videos=(\d+)`
  - `audio=(\d+)`

If parse succeeds, show compact status badges next to the message.

## 2) `client/src/pages/Directory.jsx`

### Keep X mode source as x.json
Your existing call is correct:
- `GET /prompt/<project>/file?file=x.json`

### Add per-post detail loading
For each X URL in `x.json`:
1. Extract `tweet_id` from `/status/<id>`.
2. Fetch `GET /prompt/<project>/file?file=<tweet_id>/post.json`.
3. Build UI row model:
- `url`
- `tweet_id`
- `created_at`
- `images_count`, `videos_count`, `audio_count`
- asset statuses (`downloaded|failed|skipped`) from `assets`

### Display subfolder contents in UI
Show expandable sections per post:
- `post.json` link:
  - `/prompt/<project>/file?file=<tweet_id>/post.json`
- `post.txt` link:
  - `/prompt/<project>/file?file=<tweet_id>/post.txt`
- assets grouped by type:
  - images, videos, audio

For each asset entry in `post.json.assets[*]`:
- Prefer filename from `local_path` basename.
- Construct view URL as:
  - `/prompt/<project>/file?file=<tweet_id>/<type>/<basename(local_path)>`

Do not pass raw `local_path` directly to `/file` because it may include `data_dir` prefix.

### Delete action note
Current backend does not provide X-specific delete semantics via `mode=x`.
- Short term: hide/disable delete in X mode, or mark as "not supported".
- Optional backend feature (future): add `/prompt/<project>/xpost/delete?id=<tweet_id>`.

## 3) `client/src/components/Reload.jsx`

Update label/help text to reflect behavior:
- Old: "Reload documents"
- Better: "Rebuild index (clears and reloads)"

Optional: add confirmation dialog before invoking reload.

## Suggested UI Behavior

1. Upload one X URL.
2. Immediately show returned summary (`tweet_id`, media counts).
3. Refresh X list from `x.json`.
4. Lazy-load `post.json` details only when row expands.
5. Provide direct links to `post.txt` and assets.

## Minimal Implementation Order In RAGUI

1. Fix X list source in `Upload.jsx` (use `/file?file=x.json`).
2. Add expanded X row detail in `Directory.jsx` (`post.json` + assets).
3. Update reload wording in `Reload.jsx`.
4. Add upload summary parsing for better UX.

## QA Checklist

- Upload valid X URL -> row appears with tweet id and counts.
- Batch upload mixed URLs -> success and failures shown correctly.
- Expand row -> `post.json` and `post.txt` accessible.
- Asset links open files from `images/`, `videos/`, `audio/`.
- Reload -> after completion, prompts still work and X content remains retrievable.
