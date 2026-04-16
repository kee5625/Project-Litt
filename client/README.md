# Project Litt Client

Minimal Next.js frontend for legal research search integration.

## Run

```bash
npm install
npm run dev
```

App runs at `http://localhost:3000`.

## Backend API

The UI calls:

- `GET /api/search`

Base URL defaults to `http://localhost:8000`.

Override with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Scope

Implemented:

- Search form with legal filters
- Request to backend search route
- Result cards with citation, court, year, score bar, tags, source link

Not implemented yet:

- `/api/research` pipeline
- case summarization generation
- add-to-draft action behavior
