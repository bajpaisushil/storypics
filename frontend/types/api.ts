/**
 * TypeScript mirrors of the backend Pydantic models (backend/app/schemas/*.py).
 * Keep in sync with the API; the OpenAPI schema is served at {API}/openapi.json.
 */

export type StyleId =
  | "cartoon"
  | "realistic"
  | "anime"
  | "watercolor"
  | "3d"
  | "storybook"
  | "digital_painting"
  | "comic";
export type AspectRatio = "1:1" | "4:3" | "16:9" | "9:16";
export type SceneCountChoice = number | "auto";
export type JobStatus =
  | "queued"
  | "generating"
  | "completed"
  | "partial"
  | "failed"
  | "cancelled"
  | "interrupted";
export type TaskStatus =
  | "pending"
  | "queued"
  | "generating"
  | "completed"
  | "failed"
  | "cancelled"
  | "skipped";
export type VersionMode = "initial" | "keep_composition" | "new_variation" | "retry" | "upload";
export type CharacterCategory = "human" | "animal" | "creature" | "object" | "group";
export type Gender = "female" | "male" | "unknown";

export interface ImageScores {
  alignment: number | null;
  quality: number | null;
  /** character id → identity similarity to its reference image */
  consistency: Record<string, number>;
  overall: number | null;
}

export interface GenerationInfo {
  seed: number;
  steps: number;
  guidance_scale: number;
  width: number;
  height: number;
  profile: string;
  device: string;
  model: string;
  reference_mode: "none" | "single" | "masked" | "unmasked";
  duration_s: number;
  peak_memory_gb: number | null;
}

export interface ImageVersion {
  id: string;
  /** root-relative backend path, e.g. /media/<job>/scene_01_v1.png */
  url: string;
  thumb_url: string;
  /** full-size WebP for on-screen viewing (url is the lossless PNG used for downloads) */
  preview_url?: string | null;
  file: string;
  created_at: string;
  mode: VersionMode;
  seed: number;
  prompt: string;
  prompt_2: string | null;
  negative_prompt: string;
  instructions: string | null;
  scores: ImageScores | null;
  duration_s: number;
  attempts: number;
  generation: Partial<GenerationInfo>;
}

export interface TaskProgress {
  kind: "anchor" | "scene";
  /** character id or scene index */
  target: string;
  label: string;
  step: number;
  total_steps: number;
  attempt: number;
}

export interface CharacterMention {
  text: string;
  start: number;
  end: number;
  unit_index: number;
  kind: "name" | "nominal" | "pronoun";
  source: "tagger" | "rules" | "resolver";
}

export interface CharacterProfile {
  id: string;
  name: string;
  species: string;
  category: CharacterCategory;
  gender: Gender;
  attributes: Record<string, string>;
  explicit_traits: string[];
  personality: string[];
  descriptor: string;
  mention_count: number;
  mentions: CharacterMention[];
  scene_indices: number[];
  is_recurring: boolean;
  reference_upload_id: string | null;
}

export interface CharacterState extends CharacterProfile {
  anchor_status: TaskStatus;
  anchor_versions: ImageVersion[];
  selected_anchor: string | null;
  anchor_error: string | null;
}

export interface SceneLocation {
  name: string;
  description: string;
  carried_over: boolean;
}

export interface SceneAction {
  actor_ids: string[];
  verb: string;
  phrase: string;
  short_phrase: string;
  target_ids: string[];
  subject_text: string | null;
}

export interface Scene {
  index: number;
  title: string;
  text: string;
  summary: string;
  unit_indices: number[];
  character_ids: string[];
  actions: SceneAction[];
  location: SceneLocation | null;
  time_of_day: string | null;
  weather: string | null;
  emotion: string;
  mood: string;
  composition: string;
  visual_description: string;
  previous_context: string | null;
  boundary_confidence: number | null;
}

export interface SceneState extends Scene {
  status: TaskStatus;
  versions: ImageVersion[];
  selected_version: string | null;
  error: string | null;
}

export interface JobSettings {
  style: StyleId;
  aspect_ratio: AspectRatio;
  scene_count: SceneCountChoice;
  seed: number;
  width: number;
  height: number;
  profile: string;
}

export interface AnalysisMeta {
  segmenter: string;
  tagger: string;
  unit_count: number;
  word_count: number;
  requested_scenes: number | "auto";
  timings_ms: Record<string, number>;
}

export interface Job {
  id: string;
  created_at: string;
  updated_at: string;
  status: JobStatus;
  title: string;
  story: string;
  settings: JobSettings;
  characters: CharacterState[];
  scenes: SceneState[];
  current: TaskProgress | null;
  warnings: string[];
  error: string | null;
  analysis: AnalysisMeta;
  /** computed by the backend: finished tasks / all tasks, 0–1 */
  progress: number;
}

export interface JobSummary {
  id: string;
  title: string;
  status: JobStatus;
  style: StyleId;
  scene_count: number;
  created_at: string;
  cover_url: string | null;
}

export interface ReferenceAssignment {
  upload_id: string;
  character_name: string;
}

export interface CreateJobRequest {
  story: string;
  style: StyleId;
  scene_count: SceneCountChoice;
  aspect_ratio: AspectRatio;
  references: ReferenceAssignment[];
  seed?: number | null;
  title?: string | null;
}

export interface RegenerateSceneRequest {
  instructions?: string | null;
  mode: "keep_composition" | "new_variation";
}

export interface SelectVersionRequest {
  version_id: string;
}

export interface UpdateCharacterRequest {
  name?: string | null;
  descriptor?: string | null;
  attributes?: Record<string, string> | null;
  regenerate?: boolean;
}

export interface UploadResponse {
  upload_id: string;
  url: string;
  width: number;
  height: number;
}

export interface GeneratorStatus {
  backend: "local" | "remote" | "disabled";
  available: boolean;
  loaded: boolean;
  device: string | null;
  profile: string | null;
  model: string | null;
  reason: string | null;
  /** IP-Adapter character references are active */
  reference_conditioning: boolean;
  /** non-fatal limitation, e.g. reference weights still downloading */
  notice: string | null;
}

export interface Health {
  status: string;
  generator: GeneratorStatus;
  analyzer: { segmenter: string; tagger: string };
  evaluator: Record<string, unknown>;
  queue_size: number;
}

export interface StyleInfo {
  id: StyleId;
  name: string;
  tagline: string;
  preview_url: string;
}

export type JobEvent =
  | { type: "job"; job: Job }
  | { type: "progress"; current: TaskProgress };

/** GET /api/usage — hosted compute used this month (Modal free credits). */
export interface UsageReport {
  provider: "modal" | "none";
  available: boolean;
  month: string | null;
  used_usd: number | null;
  billed_usd: number | null;
  free_credit_usd: number | null;
  remaining_usd: number | null;
  exhausted: boolean;
  resets_on: string | null;
  updated_at: string | null;
  note: string | null;
}
