import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { Job } from "../types/api.ts";
import { currentTaskLabel, jobProgress, percent, wordCount } from "../lib/format.ts";

function job(partial: Partial<Job>): Job {
  return {
    id: "j1", created_at: "", updated_at: "", status: "generating", title: "T", story: "", progress: 0,
    settings: { style: "storybook", aspect_ratio: "4:3", scene_count: "auto", seed: 1, width: 1152, height: 896, profile: "" },
    characters: [], scenes: [], current: null, warnings: [], error: null,
    analysis: { segmenter: "", tagger: "", unit_count: 0, word_count: 0, requested_scenes: "auto", timings_ms: {} },
    ...partial,
  } as Job;
}

const scene = (index: number, status: string) => ({ index, status }) as unknown as Job["scenes"][number];

describe("wordCount", () => {
  it("counts words across whitespace and newlines", () => {
    assert.equal(wordCount(""), 0);
    assert.equal(wordCount("   "), 0);
    assert.equal(wordCount("Once upon\n a  time"), 4);
  });
});

describe("percent", () => {
  it("formats scores and handles missing values", () => {
    assert.equal(percent(0.8234), "82%");
    assert.equal(percent(null), "—");
    assert.equal(percent(undefined), "—");
  });
});

describe("jobProgress", () => {
  it("counts finished tasks plus the running step fraction", () => {
    const j = job({
      scenes: [scene(1, "completed"), scene(2, "generating"), scene(3, "queued"), scene(4, "failed")],
      current: { kind: "scene", target: "2", label: "Illustrating scene 2", step: 4, total_steps: 8, attempt: 1 },
    });
    assert.equal(jobProgress(j), (2 + 0.5) / 4);
  });

  it("is 0 for a job without tasks", () => {
    assert.equal(jobProgress(job({})), 0);
  });
});

describe("currentTaskLabel", () => {
  it("adds scene totals, steps and retries", () => {
    const j = job({
      scenes: [scene(1, "completed"), scene(2, "generating"), scene(3, "queued")],
      current: { kind: "scene", target: "2", label: "Illustrating scene 2", step: 3, total_steps: 8, attempt: 2 },
    });
    assert.equal(currentTaskLabel(j), "Illustrating scene 2 of 3 · step 3/8 · attempt 2");
  });

  it("returns null when nothing is running", () => {
    assert.equal(currentTaskLabel(job({ current: null })), null);
  });
});
