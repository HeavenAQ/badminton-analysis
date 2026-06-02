import { FieldValue } from '@google-cloud/firestore'
import { createError, readBody } from 'h3'
import { getAnnotationCollection } from '~/server/utils/firestore'

interface AnnotationPayload {
  score: number | null
  feedback: string
  correction_suggestion: string
  usable_for_training: 'yes' | 'no'
  annotator: string
  notes: string
  metadata: Record<string, unknown>
}

export default defineEventHandler(async (event) => {
  const sampleId = event.context.params?.sampleId
  if (!sampleId) {
    throw createError({ statusCode: 400, statusMessage: 'Missing sample id' })
  }

  const body = await readBody<AnnotationPayload>(event)
  const docId = decodeURIComponent(sampleId)
  const payload = {
    sample_id: docId,
    score: body.score ?? null,
    feedback: body.feedback || '',
    correction_suggestion: body.correction_suggestion || '',
    usable_for_training: body.usable_for_training === 'no' ? 'no' : 'yes',
    annotator: body.annotator || '',
    notes: body.notes || '',
    metadata: body.metadata || {},
    updated_at: FieldValue.serverTimestamp()
  }

  await getAnnotationCollection().doc(docId).set(payload, { merge: true })
  return {
    ok: true,
    annotation: payload
  }
})
