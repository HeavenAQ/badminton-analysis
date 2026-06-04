import { createError, readBody } from 'h3'
import { getAnnotationDb } from '~/server/utils/firestore'

interface StatusPayload {
  sample_ids?: string[]
}

type AnnotationState = 'unannotated' | 'in_progress' | 'annotated'

const MAX_BATCH_SIZE = 300
const MAX_REQUEST_SIZE = 1000

function annotationState(data: Record<string, unknown> | undefined): AnnotationState {
  if (!data) return 'unannotated'

  const hasScore = typeof data.score === 'number'
  const hasFeedback = typeof data.feedback === 'string' && data.feedback.trim().length > 0
  const hasCorrection =
    typeof data.correction_suggestion === 'string' && data.correction_suggestion.trim().length > 0

  return hasScore && (hasFeedback || hasCorrection) ? 'annotated' : 'in_progress'
}

export default defineEventHandler(async (event) => {
  const body = await readBody<StatusPayload>(event)
  const sampleIds = [...new Set(body.sample_ids || [])]

  if (!sampleIds.length) {
    return { statuses: {} as Record<string, AnnotationState> }
  }

  if (sampleIds.length > MAX_REQUEST_SIZE) {
    throw createError({
      statusCode: 400,
      statusMessage: `Too many sample ids. Send at most ${MAX_REQUEST_SIZE} per request.`
    })
  }

  const db = getAnnotationDb()
  const config = useRuntimeConfig()
  const statuses: Record<string, AnnotationState> = {}

  for (let start = 0; start < sampleIds.length; start += MAX_BATCH_SIZE) {
    const batch = sampleIds.slice(start, start + MAX_BATCH_SIZE)
    const refs = batch.map((sampleId) => db.collection(config.firestoreCollection).doc(sampleId))
    const snapshots = await db.getAll(...refs)

    snapshots.forEach((snapshot, index) => {
      statuses[batch[index]] = annotationState(snapshot.exists ? snapshot.data() : undefined)
    })
  }

  return { statuses }
})
