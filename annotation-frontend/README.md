# Badminton Annotation Frontend

Nuxt app for expert annotation of generated `llm-annotations` samples.

## Setup

Use Node 22 LTS. This is the same runtime version configured for Netlify.

```bash
nvm use
```

1. Create a Firestore database for annotations, for example `badminton-annotations`.
2. Copy `.env.example` to `.env` and keep the server-side project/database settings.
3. Make sure local Application Default Credentials can access Firestore:

```bash
gcloud auth application-default login
```

4. Install and run:

```bash
npm install
npm run dev
```

The app reads `../llm-annotations/annotation_template.jsonl` by default and serves images from `../llm-annotations`.

For Netlify, set `FIRESTORE_SERVICE_ACCOUNT_JSON` or `FIRESTORE_SERVICE_ACCOUNT_BASE64`
as a secret environment variable in the Netlify UI.

Firestore documents are saved server-side in `FIRESTORE_COLLECTION`, keyed by `sample_id`.
No Firebase web API keys are exposed to the browser.
