import { createError } from 'h3'
import { getAnnotationCollection } from '~/server/utils/firestore'

export default defineEventHandler(async (event) => {
  const sampleId = event.context.params?.sampleId
  if (!sampleId) {
    throw createError({ statusCode: 400, statusMessage: 'Missing sample id' })
  }

  const snap = await getAnnotationCollection().doc(decodeURIComponent(sampleId)).get()
  return {
    exists: snap.exists,
    annotation: snap.exists ? snap.data() : null
  }
})
