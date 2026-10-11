"""A failed model transfer must never be advertised as an installed GGUF."""

from contextlib import contextmanager

import pytest

from evolora.local import models


class Response:
    def __init__(self, status, headers, chunks):
        self.status_code = status
        self.headers = headers
        self.chunks = chunks

    def raise_for_status(self):
        assert self.status_code in (200, 206)

    def iter_bytes(self, size):
        yield from self.chunks


def transfer(monkeypatch, tmp_path, response, partial=b""):
    model = models.CATALOG[models.DEFAULT_PLANNER]
    target = models.model_path(tmp_path, model)
    part = target.with_suffix(target.suffix + ".part")
    if partial:
        part.write_bytes(partial)

    @contextmanager
    def stream(*args, **kwargs):
        yield response

    monkeypatch.setattr(models.httpx, "stream", stream)
    return model, target, part


@pytest.mark.parametrize("size", [3, 5])
def test_rejected_range_cannot_promote_wrong_size(monkeypatch, tmp_path, size):
    model, target, part = transfer(
        monkeypatch, tmp_path, Response(416, {"content-range": "bytes */4"}, []), b"x" * size
    )
    with pytest.raises(RuntimeError, match="partial size"):
        models.download(model, tmp_path)
    assert not target.exists()
    assert part.stat().st_size == size


def test_exact_complete_partial_can_be_promoted(monkeypatch, tmp_path):
    model, target, part = transfer(
        monkeypatch, tmp_path, Response(416, {"content-range": "bytes */4"}, []), b"done"
    )
    assert models.download(model, tmp_path) == target
    assert target.read_bytes() == b"done"
    assert not part.exists()


def test_wrong_resumed_offset_preserves_partial(monkeypatch, tmp_path):
    model, target, part = transfer(
        monkeypatch, tmp_path,
        Response(206, {"content-range": "bytes 0-3/4", "content-length": "4"}, [b"done"]),
        b"do",
    )
    with pytest.raises(RuntimeError, match="unexpected byte range"):
        models.download(model, tmp_path)
    assert not target.exists()
    assert part.read_bytes() == b"do"


def test_short_transfer_is_not_installed(monkeypatch, tmp_path):
    model, target, part = transfer(
        monkeypatch, tmp_path, Response(200, {"content-length": "4"}, [b"do"])
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        models.download(model, tmp_path)
    assert not target.exists()
    assert part.read_bytes() == b"do"


def test_resumed_transfer_checks_total_and_installs(monkeypatch, tmp_path):
    model, target, part = transfer(
        monkeypatch, tmp_path,
        Response(206, {"content-range": "bytes 2-3/4", "content-length": "2"}, [b"ne"]),
        b"do",
    )
    assert models.download(model, tmp_path) == target
    assert target.read_bytes() == b"done"
    assert not part.exists()


def test_ignored_range_replaces_partial(monkeypatch, tmp_path):
    model, target, part = transfer(
        monkeypatch, tmp_path, Response(200, {"content-length": "4"}, [b"done"]), b"old"
    )
    models.download(model, tmp_path)
    assert target.read_bytes() == b"done"
    assert not part.exists()
