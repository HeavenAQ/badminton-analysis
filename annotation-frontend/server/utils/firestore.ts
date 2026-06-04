import { Firestore } from '@google-cloud/firestore'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

let db: Firestore | null = null

function readCredentialJson() {
  if (process.env.FIRESTORE_SERVICE_ACCOUNT_JSON) {
    return process.env.FIRESTORE_SERVICE_ACCOUNT_JSON
  }

  if (process.env.FIRESTORE_SERVICE_ACCOUNT_BASE64) {
    return Buffer.from(process.env.FIRESTORE_SERVICE_ACCOUNT_BASE64, 'base64').toString('utf-8')
  }

  const credentialPath = process.env.FIRESTORE_SERVICE_ACCOUNT_PATH || process.env.GOOGLE_APPLICATION_CREDENTIALS
  if (!credentialPath) {
    return ''
  }

  const resolvedPath = resolve(process.cwd(), credentialPath)
  if (!existsSync(resolvedPath)) {
    throw new Error(
      `Firestore credential file was not found at ${resolvedPath}. Set FIRESTORE_SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS to a readable service account JSON file.`
    )
  }

  return readFileSync(resolvedPath, 'utf-8')
}

export function getAnnotationDb() {
  const config = useRuntimeConfig()
  if (!db) {
    const serviceAccountJson = readCredentialJson()
    const credentials = serviceAccountJson ? JSON.parse(serviceAccountJson) : undefined

    db = new Firestore({
      projectId: config.gcpProjectId,
      databaseId: config.firestoreDatabaseId,
      credentials
    })
  }
  return db
}

export function getAnnotationCollection() {
  const config = useRuntimeConfig()
  return getAnnotationDb().collection(config.firestoreCollection)
}
