import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { API_URL, downloads, formatDetail, mediaUrl } from "../lib/api.ts";

describe("formatDetail", () => {
  it("passes through string details", () => {
    assert.equal(formatDetail("Server busy, try again shortly.", "fallback"), "Server busy, try again shortly.");
  });

  it("flattens FastAPI validation errors", () => {
    const detail = [{ loc: ["body", "story"], msg: "String should have at least 20 characters" }];
    assert.equal(formatDetail(detail, "fallback"), "story: String should have at least 20 characters");
  });

  it("falls back when there is no detail", () => {
    assert.equal(formatDetail(undefined, "Request failed (500)"), "Request failed (500)");
  });
});

describe("urls", () => {
  it("prefixes backend media paths and keeps absolute URLs", () => {
    assert.equal(mediaUrl("/media/j1/scene_01_v1.png"), `${API_URL}/media/j1/scene_01_v1.png`);
    assert.equal(mediaUrl("https://cdn.example/x.png"), "https://cdn.example/x.png");
    assert.equal(mediaUrl(null), undefined);
  });

  it("builds download links for scenes, versions and characters", () => {
    assert.equal(downloads.sceneImage("j1", 2), `${API_URL}/api/jobs/j1/scenes/2/download`);
    assert.equal(downloads.sceneImage("j1", 2, "v2"), `${API_URL}/api/jobs/j1/scenes/2/download?version=v2`);
    assert.equal(downloads.characterImage("j1", "c1"), `${API_URL}/api/jobs/j1/characters/c1/download`);
    assert.equal(downloads.zip("j1"), `${API_URL}/api/jobs/j1/export.zip`);
    assert.equal(downloads.pdf("j1"), `${API_URL}/api/jobs/j1/export.pdf`);
  });
});
