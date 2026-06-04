# Netlify Deployment

This app is prepared for Netlify with server-side Firestore writes.

## Build Settings

Use `annotation-frontend` as the Netlify base directory.

- Build command: `npm run build`
- Publish directory: `.output/public`

`netlify.toml` in this directory already matches these settings.

## Required Environment Variables

Set these in Netlify Site configuration -> Environment variables:

- `GCP_PROJECT_ID=moe-linebot-2025`
- `FIRESTORE_DATABASE_ID=badminton-annotations`
- `FIRESTORE_COLLECTION=badminton_vlm_annotations`
- `FIRESTORE_SERVICE_ACCOUNT_BASE64=<contents of /private/tmp/badminton-annotation-netlify-key.base64.txt>`

Do not expose Firebase browser keys. This app does not need them.

## Setting Secrets With Netlify CLI

The service account key was generated locally outside the repo:

```bash
/private/tmp/badminton-annotation-netlify-key.json
/private/tmp/badminton-annotation-netlify-key.base64.txt
```

Log in and link this frontend directory to your Netlify site:

```bash
netlify login
cd annotation-frontend
netlify link
```

Then set the environment variables:

```bash
netlify env:set GCP_PROJECT_ID moe-linebot-2025
netlify env:set FIRESTORE_DATABASE_ID badminton-annotations
netlify env:set FIRESTORE_COLLECTION badminton_vlm_annotations
netlify env:set FIRESTORE_SERVICE_ACCOUNT_BASE64 "$(cat /private/tmp/badminton-annotation-netlify-key.base64.txt)"
```

Verify:

```bash
netlify env:list
```

## Static Annotation Assets

After frame extraction finishes, prepare static assets:

```bash
cd annotation-frontend
npm run prepare:assets
```

This copies:

- `llm-annotations/annotation_template.jsonl` -> `annotation-frontend/public/annotation_template.jsonl`
- `llm-annotations/annotation_template.csv` -> `annotation-frontend/public/annotation_template.csv`
- `llm-annotations/annotation_schema.json` -> `annotation-frontend/public/annotation_schema.json`
- `llm-annotations/<skill>/...jpg` -> `annotation-frontend/public/annotation-images/<skill>/...jpg`

For Git-based Netlify deploys, commit the prepared `public` annotation assets.
For manual Netlify CLI deploys, prepare assets locally before deploy.
