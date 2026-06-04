# Firestore Setup

Created database:

- Project: `moe-linebot-2025`
- Database ID: `badminton-annotations`
- Location: `nam5`
- Type: Firestore Native

Copy server-side settings into `annotation-frontend/.env`:

```bash
cp annotation-frontend/.env.example annotation-frontend/.env
```

Required server config keys:

- `GCP_PROJECT_ID=moe-linebot-2025`
- `FIRESTORE_DATABASE_ID=badminton-annotations`
- `FIRESTORE_COLLECTION=badminton_vlm_annotations`

For local dev, keep the service account JSON outside the repo and point to it:

- `FIRESTORE_SERVICE_ACCOUNT_PATH=/private/tmp/badminton-annotation-netlify-key.json`

The app also supports `GOOGLE_APPLICATION_CREDENTIALS` if you already use that for local Google Cloud tools.

For Netlify, add one secret credential env var in the Netlify UI:

- `FIRESTORE_SERVICE_ACCOUNT_JSON` with the full service account JSON, or
- `FIRESTORE_SERVICE_ACCOUNT_BASE64` with the base64 encoded service account JSON.

The app writes one document per `sample_id` through Nuxt server API routes.
No Firebase browser SDK or public Firebase key is used.
