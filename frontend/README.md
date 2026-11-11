# StoryPix frontend

Next.js 16 (App Router) · TypeScript · Tailwind CSS v4 · shadcn/ui (Radix) · lucide-react.

The frontend is a static shell. All AI work happens in the StoryPix FastAPI backend running on your own machine.

## Run

```bash
npm install
cp .env.example .env.local   # optional; defaults to http://localhost:8000
npm run dev                  # http://localhost:3000
```

Start the backend first (`cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000`).
If it isn't reachable, the app shows setup instructions and retries every 5 seconds.

| Script | What it does |
|---|---|
| `npm run dev` | development server |
| `npm run build` / `npm start` | production build (all routes prerender as static) |
| `npm run lint` | ESLint (next/core-web-vitals + TypeScript) |
| `npm run typecheck` | route type generation + `tsc --noEmit` |
| `npm test` | unit tests for the pure helpers (Node's built-in test runner) |

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | StoryPix backend. Its CORS settings must allow this frontend's origin. |

## Structure

```
app/          layout, / (studio), /stories (library), error + not-found pages
components/   studio, story-form, style-picker, reference-uploader, option-selectors,
              story-viewer, progress-header, scene-preview, thumbnail-strip, version-compare,
              prompt-details, regenerate-dialog, cast-panel, edit-character-dialog,
              download-menu, generator-status-banner, backend-setup-panel, error-boundary, ui/*
hooks/        use-job (SSE + backoff + polling fallback), use-health, use-styles
lib/          api.ts (typed client, download URLs), format.ts, styles.ts
types/api.ts  TypeScript mirrors of the backend Pydantic models
tests/        unit tests
```

Style cards load `public/styles/<style-id>.webp` sample images produced by the real pipeline. Until those
files exist, each card shows a neutral colour swatch instead of an image, never a fake sample.
