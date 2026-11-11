import io
import zipfile

from PIL import Image

from tests.conftest import HEN_STORY, FakeGenerator, wait_for


def test_health_and_styles(make_client):
    client, _ = make_client()
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["generator"]["available"] is True
    styles = client.get("/api/styles").json()
    assert {s["id"] for s in styles} == {
        "cartoon",
        "realistic",
        "anime",
        "watercolor",
        "3d",
        "storybook",
        "digital_painting",
        "comic",
    }


def test_analyze_returns_scenes_and_characters(make_client):
    client, _ = make_client()
    r = client.post("/api/story/analyze", json={"story": HEN_STORY, "scene_count": 5})
    assert r.status_code == 200
    data = r.json()
    assert [s["title"] for s in data["scenes"]][1] == "The Hen Meets the Cow"
    assert len(data["scenes"]) == 5
    species = {c["species"] for c in data["characters"]}
    assert {"hen", "cow", "rabbit"} <= species


def test_validation_errors(make_client):
    client, _ = make_client()
    assert client.post("/api/jobs", json={"story": "too short"}).status_code == 422
    assert client.post("/api/jobs", json={"story": HEN_STORY, "style": "pixel"}).status_code == 422
    assert client.post("/api/jobs", json={"story": HEN_STORY, "scene_count": 40}).status_code == 422
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_full_job_lifecycle_and_exports(make_client):
    client, gen = make_client()
    r = client.post(
        "/api/jobs",
        json={"story": HEN_STORY, "style": "watercolor", "scene_count": 5, "aspect_ratio": "16:9", "seed": 42},
    )
    assert r.status_code == 202, r.text
    job = wait_for(client, r.json()["id"])
    assert job["status"] == "completed"
    kinds = [q.kind for q in gen.requests]
    # anchors for recurring characters are drawn before any scene
    assert kinds[: kinds.index("scene")] == ["anchor"] * kinds.index("scene") and kinds.count("scene") == 5
    scene_reqs = [q for q in gen.requests if q.kind == "scene"]
    assert all("watercolor painting" in q.prompt for q in scene_reqs)
    assert (scene_reqs[0].width, scene_reqs[0].height) == (1344, 768)
    # multi-character scenes are conditioned on several anchors, each confined to a region
    assert any(len(q.references) == 2 and all(ref.region for ref in q.references) for q in scene_reqs)
    for scene in job["scenes"]:
        assert scene["selected_version"] == "v1"
        assert client.get(scene["versions"][0]["url"]).status_code == 200
        preview = client.get(scene["versions"][0]["preview_url"])
        assert preview.status_code == 200 and preview.content[8:12] == b"WEBP"  # light image for the viewer

    z = client.get(f"/api/jobs/{job['id']}/export.zip")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "storybook.json" in names and sum(n.endswith(".png") and "/" not in n for n in names) == 5
    pdf = client.get(f"/api/jobs/{job['id']}/export.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")


def test_regenerate_scene_keeps_history(make_client):
    client, gen = make_client()
    job = wait_for(client, client.post("/api/jobs", json={"story": HEN_STORY, "scene_count": 3}).json()["id"])
    first_seed = job["scenes"][1]["versions"][0]["seed"]
    r = client.post(
        f"/api/jobs/{job['id']}/scenes/2/regenerate",
        json={"instructions": "Make the hen happier", "mode": "keep_composition"},
    )
    assert r.status_code == 200
    job = wait_for(client, job["id"])
    scene = job["scenes"][1]
    assert [v["id"] for v in scene["versions"]] == ["v1", "v2"] and scene["selected_version"] == "v2"
    assert scene["versions"][1]["seed"] == first_seed  # keep composition → same seed
    assert "very happy" in gen.requests[-1].prompt
    r = client.post(f"/api/jobs/{job['id']}/scenes/2/select", json={"version_id": "v1"})
    assert r.json()["scenes"][1]["selected_version"] == "v1"


def test_failed_scenes_are_reported_and_resumable(make_client):
    failing = FakeGenerator(fail_kinds={"scene"})
    client, _ = make_client(failing)
    job = wait_for(client, client.post("/api/jobs", json={"story": HEN_STORY, "scene_count": 3}).json()["id"])
    assert job["status"] == "failed"
    assert all(s["status"] == "failed" and "simulated" in s["error"] for s in job["scenes"])
    failing.fail_kinds.clear()
    client.post(f"/api/jobs/{job['id']}/resume")
    assert wait_for(client, job["id"])["status"] == "completed"


def test_unavailable_generator_returns_503(make_client):
    class Offline(FakeGenerator):
        def status(self):
            s = super().status()
            return s.model_copy(update={"available": False, "reason": "weights missing"})

    client, _ = make_client(Offline())
    r = client.post("/api/jobs", json={"story": HEN_STORY})
    assert r.status_code == 503 and "weights missing" in r.json()["detail"]


def test_reference_upload_is_assigned_to_character(make_client):
    client, gen = make_client()
    buf = io.BytesIO()
    Image.new("RGB", (300, 300), (10, 200, 30)).save(buf, format="PNG")
    up = client.post("/api/uploads", files={"file": ("hen.png", buf.getvalue(), "image/png")})
    assert up.status_code == 200
    bad = client.post("/api/uploads", files={"file": ("x.png", b"not an image", "image/png")})
    assert bad.status_code == 400
    r = client.post(
        "/api/jobs",
        json={
            "story": HEN_STORY,
            "scene_count": 3,
            "references": [{"upload_id": up.json()["upload_id"], "character_name": "hen"}],
        },
    )
    job = wait_for(client, r.json()["id"])
    hen = next(c for c in job["characters"] if c["species"] == "hen")
    assert hen["reference_upload_id"] == up.json()["upload_id"]
    anchor_req = next(q for q in gen.requests if q.kind == "anchor" and q.references)
    assert anchor_req.references[0].character_id == hen["id"]


def test_cancel(make_client):
    client, _ = make_client(FakeGenerator(delay=0.05))
    job = client.post("/api/jobs", json={"story": HEN_STORY, "scene_count": 5}).json()
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.json()["status"] == "cancelled"
    final = client.get(f"/api/jobs/{job['id']}").json()
    assert final["status"] == "cancelled"


def test_image_downloads_are_attachments(make_client):
    client, _ = make_client()
    job = wait_for(client, client.post("/api/jobs", json={"story": HEN_STORY, "scene_count": 3}).json()["id"])
    r = client.get(f"/api/jobs/{job['id']}/scenes/1/download")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert 'attachment; filename="01-the-hen-on-the-farm.png"' in r.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job['id']}/scenes/1/download?version=v9").status_code == 400
    hen = next(c for c in job["characters"] if c["species"] == "hen")
    r = client.get(f"/api/jobs/{job['id']}/characters/{hen['id']}/download")
    assert r.status_code == 200 and "character-hen.png" in r.headers["content-disposition"]


def test_access_key_and_job_cap(settings, analyzer):
    from fastapi.testclient import TestClient

    from app.main import create_app

    locked = settings.model_copy(update={"access_key": "s3cret", "max_active_jobs": 1})
    with TestClient(create_app(locked, generator=FakeGenerator(delay=0.2), analyzer=analyzer)) as client:
        assert client.get("/api/health").json()["requires_key"] is True
        assert client.post("/api/story/analyze", json={"story": HEN_STORY}).status_code == 200  # read-only, open
        assert client.post("/api/jobs", json={"story": HEN_STORY}).status_code == 401
        headers = {"X-StoryPix-Key": "s3cret"}
        assert client.post("/api/jobs", json={"story": HEN_STORY}, headers=headers).status_code == 202
        busy = client.post("/api/jobs", json={"story": HEN_STORY}, headers=headers)
        assert busy.status_code == 429 and "busy" in busy.json()["detail"]


def test_hosted_frontend_can_reach_local_backend(settings, analyzer):
    from fastapi.testclient import TestClient

    from app.main import create_app

    cfg = settings.model_copy(update={"cors_origin_regex": r"https://storypics.*\.vercel\.app"})
    with TestClient(create_app(cfg, generator=FakeGenerator(), analyzer=analyzer)) as client:
        r = client.options(
            "/api/jobs",
            headers={
                "Origin": "https://storypics.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        assert r.headers["access-control-allow-origin"] == "https://storypics.vercel.app"
        assert r.headers["access-control-allow-private-network"] == "true"
        evil = client.options(
            "/api/jobs", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"}
        )
        assert "access-control-allow-origin" not in evil.headers


def test_usage_endpoint_outside_modal(make_client):
    client, _ = make_client()
    r = client.get("/api/usage").json()
    assert r["provider"] == "none" and r["available"] is False
