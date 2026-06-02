<template>
  <main class="page">
    <header class="topbar">
      <div>
        <h1>Badminton Expert Annotation</h1>
        <p>{{ statusText }}</p>
      </div>
      <div class="top-actions">
        <input v-model.trim="annotator" class="annotator" placeholder="Annotator name" />
        <button @click="reload" :disabled="pending">Reload</button>
      </div>
    </header>

    <section class="filters">
      <label>
        Skill
        <select v-model="filters.skill" @change="reload">
          <option value="">All</option>
          <option v-for="skill in facets?.skills || []" :key="skill" :value="skill">{{ skill }}</option>
        </select>
      </label>
      <label>
        Key Frame
        <select v-model="filters.keyFrame" @change="reload">
          <option value="">All</option>
          <option v-for="keyFrame in facets?.keyFrames || []" :key="keyFrame" :value="keyFrame">
            {{ keyFrame }}
          </option>
        </select>
      </label>
      <label>
        Cohort
        <select v-model="filters.cohort" @change="reload">
          <option value="">All</option>
          <option v-for="cohort in facets?.cohorts || []" :key="cohort" :value="cohort">{{ cohort }}</option>
        </select>
      </label>
      <label>
        Source
        <select v-model="filters.sourceDataset" @change="reload">
          <option value="">All</option>
          <option v-for="source in facets?.sourceDatasets || []" :key="source" :value="source">
            {{ source }}
          </option>
        </select>
      </label>
    </section>

    <section v-if="loadError" class="message error">{{ loadError }}</section>
    <section v-else-if="pending" class="message">Loading samples...</section>
    <section v-else-if="!currentSample" class="message">No samples match these filters.</section>

    <section v-else class="workspace">
      <aside class="sample-list">
        <button
          v-for="(sample, index) in samples"
          :key="sample.sample_id"
          :class="{ active: index === currentIndex }"
          @click="selectSample(index)"
        >
          <span>{{ index + 1 + offset }}</span>
          <strong>{{ sample.metadata.skill }} / KF{{ sample.metadata.key_frame_index }}</strong>
          <small>{{ sample.metadata.video_file }} · {{ sample.metadata.neighbor_offset }}</small>
        </button>
      </aside>

      <section class="viewer">
        <div class="image-wrap">
          <img :src="imageUrl(currentSample.image)" :alt="currentSample.sample_id" />
        </div>
        <div class="meta-grid">
          <span>Skill: <strong>{{ currentSample.metadata.skill }}</strong></span>
          <span>Key frame: <strong>{{ currentSample.metadata.key_frame_name }}</strong></span>
          <span>Offset: <strong>{{ currentSample.metadata.neighbor_offset }}</strong></span>
          <span>Frame: <strong>{{ currentSample.metadata.frame_index }}</strong></span>
          <span>Hand: <strong>{{ currentSample.metadata.handedness }}</strong></span>
          <span>Source: <strong>{{ currentSample.metadata.source_dataset }}</strong></span>
        </div>
        <details class="angles">
          <summary>Angles metadata</summary>
          <div>
            <span v-for="(value, name) in currentSample.metadata.angles" :key="name">
              {{ name }}: <strong>{{ Number(value).toFixed(1) }}</strong>
            </span>
          </div>
        </details>
      </section>

      <form class="annotation" @submit.prevent="saveCurrent">
        <h2>Expert Label</h2>
        <label>
          Score
          <input v-model.number="form.score" type="number" min="0" max="10" step="0.5" placeholder="0-10" />
        </label>
        <label>
          Feedback
          <textarea v-model.trim="form.feedback" rows="5" placeholder="What is technically correct or incorrect?"></textarea>
        </label>
        <label>
          Correction suggestion
          <textarea v-model.trim="form.correction_suggestion" rows="4" placeholder="Actionable coaching cue"></textarea>
        </label>
        <label>
          Use for training
          <select v-model="form.usable_for_training">
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </label>
        <label>
          Notes
          <textarea v-model.trim="form.notes" rows="3" placeholder="Occlusion, tracking issue, ambiguity"></textarea>
        </label>

        <div class="form-actions">
          <button type="button" @click="previousSample" :disabled="currentIndex === 0">Previous</button>
          <button type="submit" class="primary" :disabled="saving">{{ saving ? 'Saving...' : 'Save & Next' }}</button>
          <button type="button" @click="nextSample" :disabled="currentIndex >= samples.length - 1">Skip</button>
        </div>
        <p v-if="saveStatus" class="save-status">{{ saveStatus }}</p>
      </form>
    </section>
  </main>
</template>

<script setup lang="ts">
import type { AnnotationSample, SavedAnnotation } from '~/types/annotation'

const PAGE_LIMIT = 200

const annotator = useCookie('badminton_annotator', { default: () => '' })
const offset = ref(0)
const currentIndex = ref(0)
const allSamples = ref<AnnotationSample[]>([])
const samples = ref<AnnotationSample[]>([])
const pending = ref(false)
const saving = ref(false)
const loadError = ref('')
const saveStatus = ref('')
const filters = reactive({
  skill: '',
  keyFrame: '',
  cohort: '',
  sourceDataset: ''
})

const form = reactive({
  score: null as number | null,
  feedback: '',
  correction_suggestion: '',
  usable_for_training: 'yes' as 'yes' | 'no',
  notes: ''
})

const facets = ref<{
  total: number
  skills: string[]
  keyFrames: string[]
  cohorts: string[]
  sourceDatasets: string[]
} | null>(null)

const currentSample = computed(() => samples.value[currentIndex.value] || null)
const statusText = computed(() => {
  const total = facets.value?.total ? `${facets.value.total} samples` : 'Waiting for manifest'
  return `${total} · Firestore writes go through the Nuxt server`
})

function imageUrl(path: string) {
  return `/annotation-images/${path}`
}

function resetForm() {
  form.score = null
  form.feedback = ''
  form.correction_suggestion = ''
  form.usable_for_training = 'yes'
  form.notes = ''
}

async function hydrateForm(sample: AnnotationSample) {
  resetForm()
  try {
    const response = await $fetch<{ exists: boolean; annotation: SavedAnnotation | null }>(
      `/api/annotations/${encodeURIComponent(sample.sample_id)}`
    )
    const saved = response.annotation
    if (!saved) return
    form.score = saved.score
    form.feedback = saved.feedback || ''
    form.correction_suggestion = saved.correction_suggestion || ''
    form.usable_for_training = saved.usable_for_training || 'yes'
    form.notes = saved.notes || ''
    annotator.value = saved.annotator || annotator.value
  } catch (error) {
    saveStatus.value = error instanceof Error ? error.message : 'Could not load saved annotation.'
  }
}

async function reload() {
  pending.value = true
  loadError.value = ''
  saveStatus.value = ''
  offset.value = 0
  currentIndex.value = 0
  try {
    samples.value = filteredSamples().slice(0, PAGE_LIMIT)
    if (samples.value[0]) await hydrateForm(samples.value[0])
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : 'Could not load samples.'
  } finally {
    pending.value = false
  }
}

function filteredSamples() {
  return allSamples.value.filter((sample) => {
    const meta = sample.metadata
    return (
      (!filters.skill || meta.skill === filters.skill) &&
      (!filters.keyFrame || meta.key_frame_name === filters.keyFrame) &&
      (!filters.cohort || meta.cohort === filters.cohort) &&
      (!filters.sourceDataset || meta.source_dataset === filters.sourceDataset)
    )
  })
}

async function loadStaticManifest() {
  pending.value = true
  loadError.value = ''
  try {
    const text = await $fetch<string>('/annotation_template.jsonl', {
      responseType: 'text'
    })
    const parsed = text
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as AnnotationSample)

    allSamples.value = parsed
    facets.value = {
      total: parsed.length,
      skills: [...new Set(parsed.map((sample) => sample.metadata.skill))].sort(),
      keyFrames: [...new Set(parsed.map((sample) => sample.metadata.key_frame_name))].sort(),
      cohorts: [...new Set(parsed.map((sample) => sample.metadata.cohort))].sort(),
      sourceDatasets: [...new Set(parsed.map((sample) => sample.metadata.source_dataset))].sort()
    }
  } catch (error) {
    loadError.value =
      error instanceof Error
        ? error.message
        : 'Could not load /annotation_template.jsonl. Run prepare:assets before deployment.'
  } finally {
    pending.value = false
  }
}

async function selectSample(index: number) {
  currentIndex.value = index
  saveStatus.value = ''
  if (currentSample.value) await hydrateForm(currentSample.value)
}

async function saveCurrent() {
  if (!currentSample.value) return
  saving.value = true
  saveStatus.value = ''
  try {
    const payload: Omit<SavedAnnotation, 'sample_id' | 'metadata' | 'updated_at'> = {
      score: form.score,
      feedback: form.feedback,
      correction_suggestion: form.correction_suggestion,
      usable_for_training: form.usable_for_training,
      annotator: annotator.value,
      notes: form.notes
    }
    await $fetch(`/api/annotations/${encodeURIComponent(currentSample.value.sample_id)}`, {
      method: 'POST',
      body: {
        ...payload,
        metadata: currentSample.value.metadata
      }
    })
    saveStatus.value = 'Saved'
    nextSample()
  } catch (error) {
    saveStatus.value = error instanceof Error ? error.message : 'Save failed.'
  } finally {
    saving.value = false
  }
}

function nextSample() {
  if (currentIndex.value < samples.value.length - 1) {
    void selectSample(currentIndex.value + 1)
  }
}

function previousSample() {
  if (currentIndex.value > 0) {
    void selectSample(currentIndex.value - 1)
  }
}

onMounted(async () => {
  await loadStaticManifest()
  await reload()
})
</script>

<style scoped>
.page {
  min-height: 100vh;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 20px;
}

.topbar p,
.sample-list small,
.meta-grid span {
  color: var(--muted);
}

.top-actions {
  display: flex;
  gap: 8px;
}

.annotator {
  width: 220px;
}

.filters {
  display: grid;
  grid-template-columns: repeat(4, minmax(140px, 1fr));
  gap: 12px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--line);
  background: #fff;
}

label {
  display: grid;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
}

.workspace {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 380px;
  gap: 16px;
  padding: 16px;
}

.sample-list {
  display: grid;
  align-content: start;
  max-height: calc(100vh - 150px);
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

.sample-list button {
  display: grid;
  grid-template-columns: 34px 1fr;
  gap: 2px 8px;
  min-height: 56px;
  border: 0;
  border-bottom: 1px solid var(--line);
  border-radius: 0;
  text-align: left;
}

.sample-list small {
  grid-column: 2;
}

.sample-list .active {
  background: var(--accent-soft);
}

.viewer,
.annotation,
.message {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

.viewer {
  padding: 12px;
}

.image-wrap {
  display: grid;
  place-items: center;
  min-height: 520px;
  background: #111827;
  border-radius: 6px;
  overflow: hidden;
}

.image-wrap img {
  max-width: 100%;
  max-height: 76vh;
  object-fit: contain;
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 12px 0 0;
  font-size: 13px;
}

.angles {
  margin-top: 10px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}

.angles summary {
  cursor: pointer;
  font-weight: 700;
}

.angles div {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px 12px;
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
}

.annotation {
  display: grid;
  align-content: start;
  gap: 14px;
  padding: 16px;
}

.form-actions {
  display: grid;
  grid-template-columns: 1fr 1.3fr 1fr;
  gap: 8px;
}

.save-status {
  color: var(--accent);
  font-weight: 700;
}

.message {
  margin: 20px;
  padding: 20px;
}

.error {
  color: var(--danger);
}

@media (max-width: 1100px) {
  .workspace {
    grid-template-columns: 1fr;
  }

  .sample-list {
    max-height: 220px;
  }

  .filters {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
