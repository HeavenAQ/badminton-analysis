import { Firestore } from '@google-cloud/firestore'

let db: Firestore | null = null

export function getAnnotationDb() {
  const config = useRuntimeConfig()
  if (!db) {
    const serviceAccountJson =
      process.env.FIRESTORE_SERVICE_ACCOUNT_JSON ||
      (process.env.FIRESTORE_SERVICE_ACCOUNT_BASE64
        ? Buffer.from(process.env.FIRESTORE_SERVICE_ACCOUNT_BASE64, 'base64').toString('utf-8')
        : '')
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
