"""Test-first coverage for X upload media snapshots and reload rebuild semantics."""

import json
from pathlib import Path

from flask import Flask

from rag.api.routes import rag_bp
from rag.services.document_loader.x_loader import XLoaderStrategy
from rag.services.vector_store import VectorStoreService


class _FakeVectorStore:
    """Minimal vectorstore stub that captures added documents."""

    def __init__(self):
        self.added_documents = []

    def add_documents(self, docs):
        self.added_documents.extend(docs)


def _write_snapshot(base_dir: Path, tweet_id: str, text: str = "hello from snapshot"):
    """Create a minimal local X snapshot under data/<tweet_id>."""
    post_dir = base_dir / tweet_id
    post_dir.mkdir(parents=True, exist_ok=True)

    (post_dir / "post.txt").write_text(text)
    (post_dir / "post.json").write_text(
        json.dumps(
            {
                "tweet_id": tweet_id,
                "post_url": f"https://x.com/user/status/{tweet_id}",
                "author": {"username": "user"},
                "indexing": {"mode": "text_only"},
            }
        )
    )


def test_x_loader_reload_uses_local_snapshots_without_api_calls(temp_dir, monkeypatch):
    """Reload mode should index from local snapshots and not call the X API."""
    loader = XLoaderStrategy()
    _write_snapshot(temp_dir, "1234567890", "snapshot content")

    # Guardrail: reload path should never reach remote fetch once implemented.
    def _forbidden_fetch(tweet_id, max_retries=3):
        raise AssertionError("reload should not call X API")

    monkeypatch.setattr(loader, "_fetch_tweet", _forbidden_fetch)

    fake_vectorstore = _FakeVectorStore()
    count = loader.load(
        {
            "data_dir": str(temp_dir),
            "chunk_size": 1000,
            "chunk_overlap": 50,
        },
        fake_vectorstore,
    )

    assert count > 0
    assert len(fake_vectorstore.added_documents) > 0


def test_vector_store_service_exposes_clear_and_rebuild_methods():
    """VectorStoreService should expose explicit clear/rebuild API for /reload."""
    assert hasattr(VectorStoreService, "clear_vectorstore")
    assert hasattr(VectorStoreService, "rebuild_documents")


def test_reload_route_rebuilds_and_reinitializes_chain(monkeypatch):
    """/reload should rebuild docs and then reinitialize chain without reloading again."""

    app = Flask(__name__)
    app.register_blueprint(rag_bp)

    class _FakeVectorStoreService:
        def __init__(self):
            self.called = False

        def rebuild_documents(self, file_type="all"):
            self.called = True
            return 1

    fake_vector_store_service = _FakeVectorStoreService()

    app.config["VECTOR_STORE_SERVICE"] = fake_vector_store_service
    app.config["RAG_CONFIG"] = object()
    app.config["APP_STATE"] = {}
    app.config["CHAT_HISTORY_SERVICE"] = object()
    app.config["CONFIG_SERVICE"] = object()

    captured = {"called": False, "new_vectorstore": None}

    def _fake_initialize_chain(new_vectorstore=False):
        captured["called"] = True
        captured["new_vectorstore"] = new_vectorstore
        return True

    monkeypatch.setattr("rag.api.routes.initialize_chain", _fake_initialize_chain)

    with app.test_client() as client:
        response = client.post("/prompt/demo/reload")

    assert response.status_code == 200
    assert fake_vector_store_service.called is True
    assert captured["called"] is True
    assert captured["new_vectorstore"] is False


def test_x_loader_upload_persists_snapshot_and_media(temp_dir, monkeypatch):
    """Upload mode should persist post.json/post.txt and download media files."""
    loader = XLoaderStrategy()

    tweet_data = {
        "data": {
            "id": "111222333",
            "author_id": "777",
            "text": "uploadx persistence test",
            "created_at": "2026-03-10T10:00:00.000Z",
            "public_metrics": {
                "retweet_count": 1,
                "like_count": 2,
                "reply_count": 3,
                "quote_count": 4,
            },
        },
        "includes": {
            "users": [{"id": "777", "username": "uploader", "name": "Uploader"}],
            "media": [
                {"type": "photo", "url": "https://cdn.example/img1.jpg"},
                {
                    "type": "video",
                    "preview_image_url": "https://cdn.example/preview.jpg",
                    "variants": [
                        {"content_type": "video/mp4", "bit_rate": 256000, "url": "https://cdn.example/v1.mp4"},
                        {"content_type": "video/mp4", "bit_rate": 512000, "url": "https://cdn.example/v2.mp4"},
                    ],
                },
                {
                    "type": "video",
                    "variants": [
                        {"content_type": "audio/mp4", "bit_rate": 64000, "url": "https://cdn.example/a1.m4a"},
                    ],
                },
            ],
        },
    }

    monkeypatch.setattr(loader, "_fetch_tweet", lambda tweet_id, max_retries=3: tweet_data)

    def _fake_download(url, target_path, timeout=30, max_bytes=None):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(b"media")
        return True

    monkeypatch.setattr(loader, "_download_asset", _fake_download)

    fake_vectorstore = _FakeVectorStore()
    url = "https://x.com/uploader/status/111222333"
    count = loader.load(
        {
            "x_urls": [url],
            "data_dir": str(temp_dir),
            "chunk_size": 1000,
            "chunk_overlap": 50,
            "download_media": True,
            "index_text_only": True,
        },
        fake_vectorstore,
    )

    assert count > 0
    post_dir = temp_dir / "111222333"
    assert (post_dir / "post.json").exists()
    assert (post_dir / "post.txt").exists()
    assert (post_dir / "images").exists()
    assert (post_dir / "videos").exists()
    assert (post_dir / "audio").exists()

    post_json = json.loads((post_dir / "post.json").read_text())
    assert post_json["tweet_id"] == "111222333"
    assert len(post_json["assets"]["images"]) == 1
    assert len(post_json["assets"]["videos"]) == 1
    assert len(post_json["assets"]["audio"]) == 1


def test_x_loader_upload_indexes_text_only_even_with_media(temp_dir, monkeypatch):
    """Upload mode should index tweet text, not binary media content."""
    loader = XLoaderStrategy()

    tweet_data = {
        "data": {
            "id": "444555666",
            "author_id": "888",
            "text": "text to embed",
            "created_at": "2026-03-10T11:00:00.000Z",
            "public_metrics": {},
        },
        "includes": {
            "users": [{"id": "888", "username": "author", "name": "Author"}],
            "media": [{"type": "photo", "url": "https://cdn.example/p.jpg"}],
        },
    }
    monkeypatch.setattr(loader, "_fetch_tweet", lambda tweet_id, max_retries=3: tweet_data)
    monkeypatch.setattr(loader, "_download_asset", lambda url, target_path, timeout=30, max_bytes=None: True)

    fake_vectorstore = _FakeVectorStore()
    count = loader.load(
        {
            "x_urls": ["https://x.com/author/status/444555666"],
            "data_dir": str(temp_dir),
            "chunk_size": 1000,
            "chunk_overlap": 50,
            "download_media": True,
            "index_text_only": True,
        },
        fake_vectorstore,
    )

    assert count > 0
    assert len(fake_vectorstore.added_documents) > 0
    assert any("text to embed" in doc.page_content for doc in fake_vectorstore.added_documents)


def test_x_loader_fetch_tweet_requests_media_expansions(monkeypatch):
    """Fetch call should request media expansions to enable local asset persistence."""
    loader = XLoaderStrategy()
    loader.api_token = "token"

    captured = {}

    class _Resp:
        status_code = 200
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {"data": {"id": "1", "text": "ok"}}

    def _fake_get(url, headers=None, params=None, timeout=0):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("rag.services.document_loader.x_loader.requests.get", _fake_get)

    result = loader._fetch_tweet("123")

    assert result is not None
    assert captured["url"].endswith("/123")
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert "article" in captured["params"]["tweet.fields"]
    assert "attachments.media_keys" in captured["params"]["expansions"]
    assert "media.fields" in captured["params"]


def test_x_loader_upload_media_failure_still_indexes_text(temp_dir, monkeypatch):
    """Media download failures should not block tweet text indexing."""
    loader = XLoaderStrategy()

    tweet_data = {
        "data": {
            "id": "999111222",
            "author_id": "777",
            "text": "index me even if media fails",
            "created_at": "2026-03-10T12:00:00.000Z",
            "public_metrics": {},
        },
        "includes": {
            "users": [{"id": "777", "username": "uploader", "name": "Uploader"}],
            "media": [{"type": "photo", "url": "https://cdn.example/fail.jpg"}],
        },
    }

    monkeypatch.setattr(loader, "_fetch_tweet", lambda tweet_id, max_retries=3: tweet_data)
    monkeypatch.setattr(loader, "_download_asset", lambda url, target_path, timeout=30, max_bytes=None: False)

    fake_vectorstore = _FakeVectorStore()
    count = loader.load(
        {
            "x_urls": ["https://x.com/uploader/status/999111222"],
            "data_dir": str(temp_dir),
            "chunk_size": 1000,
            "chunk_overlap": 50,
            "download_media": True,
            "index_text_only": True,
        },
        fake_vectorstore,
    )

    assert count > 0
    assert any("index me even if media fails" in doc.page_content for doc in fake_vectorstore.added_documents)

    post_json = json.loads((temp_dir / "999111222" / "post.json").read_text())
    assert post_json["assets"]["images"][0]["status"] == "failed"


def test_x_loader_upload_sets_snapshot_metadata_on_documents(temp_dir, monkeypatch):
    """Upload-mode indexed documents should include snapshot and media metadata."""
    loader = XLoaderStrategy()

    tweet_data = {
        "data": {
            "id": "321654987",
            "author_id": "456",
            "text": "metadata contract text",
            "created_at": "2026-03-10T13:00:00.000Z",
            "public_metrics": {},
        },
        "includes": {
            "users": [{"id": "456", "username": "metauser", "name": "Meta User"}],
            "media": [{"type": "photo", "url": "https://cdn.example/meta.jpg"}],
        },
    }

    monkeypatch.setattr(loader, "_fetch_tweet", lambda tweet_id, max_retries=3: tweet_data)
    monkeypatch.setattr(loader, "_download_asset", lambda url, target_path, timeout=30, max_bytes=None: True)

    fake_vectorstore = _FakeVectorStore()
    count = loader.load(
        {
            "x_urls": ["https://x.com/metauser/status/321654987"],
            "data_dir": str(temp_dir),
            "chunk_size": 1000,
            "chunk_overlap": 50,
            "download_media": True,
            "index_text_only": True,
        },
        fake_vectorstore,
    )

    assert count > 0
    assert len(fake_vectorstore.added_documents) > 0
    metadata = fake_vectorstore.added_documents[0].metadata
    assert metadata["source"] == "x"
    assert metadata["tweet_id"] == "321654987"
    assert metadata["author_username"] == "metauser"
    assert metadata["indexing_mode"] == "text_only"
    assert metadata["images_count"] == 1
    assert metadata["videos_count"] == 0
    assert metadata["audio_count"] == 0
    assert metadata["snapshot_path"].endswith("321654987/post.json")
    assert metadata["text_path"].endswith("321654987/post.txt")


def test_x_loader_extract_primary_text_prefers_note_and_referenced_content():
    """Primary text extraction should prefer note_tweet, expand URLs, and include referenced text."""
    loader = XLoaderStrategy()

    payload = {
        "data": {
            "text": "https://t.co/abc",
            "note_tweet": {
                "text": "Long form content https://t.co/abc"
            },
            "entities": {
                "urls": [
                    {
                        "url": "https://t.co/abc",
                        "expanded_url": "https://example.com/full-article"
                    }
                ]
            }
        },
        "includes": {
            "tweets": [
                {
                    "text": "Referenced tweet text"
                }
            ]
        }
    }

    text = loader._extract_primary_text(payload)

    assert "Long form content" in text
    assert "https://example.com/full-article" in text
    assert "Referenced post(s):" in text
    assert "Referenced tweet text" in text


def test_x_loader_extract_primary_text_uses_article_plain_text():
    """Primary extraction should use article plain text when note_tweet is absent."""
    loader = XLoaderStrategy()

    payload = {
        "data": {
            "text": "https://t.co/article",
            "article": {
                "title": "Example",
                "plain_text": "This is the full article body text.",
                "preview_text": "Short preview",
            },
            "entities": {
                "urls": [
                    {
                        "url": "https://t.co/article",
                        "expanded_url": "https://x.com/i/article/123"
                    }
                ]
            }
        },
        "includes": {}
    }

    text = loader._extract_primary_text(payload)
    assert text == "This is the full article body text."


def test_x_loader_extract_primary_text_enriches_url_only_posts(monkeypatch):
    """URL-only tweet text should be enriched with linked content preview."""
    loader = XLoaderStrategy()

    payload = {
        "data": {
            "text": "https://t.co/onlyurl",
            "entities": {
                "urls": [
                    {
                        "url": "https://t.co/onlyurl",
                        "expanded_url": "https://example.com/long"
                    }
                ]
            }
        },
        "includes": {}
    }

    monkeypatch.setattr(
        loader,
        "_fetch_link_preview_text",
        lambda url: "Title: Example Long Form\nDescription: Full text summary",
    )

    text = loader._extract_primary_text(payload)

    assert "https://example.com/long" in text
    assert "Linked content:" in text
    assert "Description: Full text summary" in text


def test_x_loader_extract_primary_text_uses_syndication_for_url_only(monkeypatch):
    """URL-only tweet text should use syndication fallback when available."""
    loader = XLoaderStrategy()

    payload = {
        "data": {
            "id": "2031801444064543156",
            "text": "https://t.co/IlSdfePEMV"
        }
    }

    monkeypatch.setattr(
        loader,
        "_fetch_syndication_text",
        lambda tweet_id: "Expanded post body from syndication"
    )

    text = loader._extract_primary_text(payload)
    assert text == "Expanded post body from syndication"
