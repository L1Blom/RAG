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


def test_reload_route_uses_rebuild_instead_of_initialize_chain(monkeypatch):
    """/reload should call vector store rebuild and avoid initialize_chain(True)."""

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

    def _forbidden_initialize_chain(new_vectorstore=False):
        raise AssertionError("reload should not call initialize_chain")

    monkeypatch.setattr("rag.api.routes.initialize_chain", _forbidden_initialize_chain)

    with app.test_client() as client:
        response = client.post("/prompt/demo/reload")

    assert response.status_code == 200
    assert fake_vector_store_service.called is True


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

    def _fake_download(url, target_path, timeout=30):
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
    monkeypatch.setattr(loader, "_download_asset", lambda url, target_path, timeout=30: True)

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
