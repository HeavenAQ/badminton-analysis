<template>
  <main class="page">
    <header class="topbar">
      <div>
        <h1>羽球專家標註系統</h1>
        <p>{{ statusText }}</p>
      </div>
      <div class="top-actions">
        <input v-model.trim="annotator" class="annotator" placeholder="標註者姓名" />
        <button @click="reload" :disabled="pending">重新載入</button>
      </div>
    </header>

    <section class="filters">
      <label>
        技術
        <select v-model="filters.skill" @change="reload">
          <option value="">全部</option>
          <option v-for="skill in facets?.skills || []" :key="skill" :value="skill">{{ skillLabel(skill) }}</option>
        </select>
      </label>
      <label>
        關鍵影格
        <select v-model="filters.keyFrame" @change="reload">
          <option value="">全部</option>
          <option v-for="keyFrame in facets?.keyFrames || []" :key="keyFrame" :value="keyFrame">
            {{ keyFrameLabel(keyFrame) }}
          </option>
        </select>
      </label>
      <label>
        組別
        <select v-model="filters.cohort" @change="reload">
          <option value="">全部</option>
          <option v-for="cohort in facets?.cohorts || []" :key="cohort" :value="cohort">{{ cohortLabel(cohort) }}</option>
        </select>
      </label>
      <label>
        資料來源
        <select v-model="filters.sourceDataset" @change="reload">
          <option value="">全部</option>
          <option v-for="source in facets?.sourceDatasets || []" :key="source" :value="source">
            {{ sourceLabel(source) }}
          </option>
        </select>
      </label>
      <label>
        標註狀態
        <select v-model="filters.annotationStatus" @change="reload">
          <option value="">全部</option>
          <option value="annotated">已完成</option>
          <option value="in_progress">標註中</option>
          <option value="unannotated">尚未標註</option>
        </select>
      </label>
    </section>

    <section v-if="loadError" class="message error">{{ loadError }}</section>
    <section v-else-if="pending" class="message loading-message">
      <span class="spinner"></span>
      <strong>{{ loadingText }}</strong>
    </section>
    <section v-else-if="!currentSample" class="message">沒有符合篩選條件的樣本。</section>

    <section v-else class="workspace">
      <div class="progress-card">
        <div>
          <span>目前樣本</span>
          <strong>{{ absolutePosition }} / {{ filteredCount }}</strong>
        </div>
        <div class="progress-track">
          <span :style="{ width: `${progressPercent}%` }"></span>
        </div>
      </div>

      <aside class="sample-list">
        <div class="sample-list-header">
          <strong>{{ pageStatus }}</strong>
          <div>
            <button type="button" @click="previousPage" :disabled="offset === 0">上一頁</button>
            <button type="button" @click="nextPage" :disabled="!hasNextPage">下一頁</button>
          </div>
        </div>
        <div class="sample-list-scroll">
          <button
            v-for="(sample, index) in samples"
            :key="sample.sample_id"
            :class="{ active: index === currentIndex }"
            @click="selectSample(index)"
          >
            <span>{{ index + 1 + offset }}</span>
            <strong>{{ skillLabel(sample.metadata.skill) }} / 關鍵影格 {{ sample.metadata.key_frame_index + 1 }}/5</strong>
            <small>
              原始影格 {{ sample.metadata.frame_index }} · 位移 {{ sample.metadata.neighbor_offset }} ·
              {{ sample.metadata.video_file }}
            </small>
            <em :class="annotationBadgeClass(sample.sample_id)">
              {{ annotationBadgeLabel(sample.sample_id) }}
            </em>
          </button>
        </div>
      </aside>

      <section class="viewer">
        <div class="frame-context">
          <div>
            <span>目前顯示影格</span>
            <strong>{{ frameTitle }}</strong>
            <small>{{ frameSubtitle }}</small>
          </div>
          <div class="frame-pills">
            <span>{{ keyFramePositionLabel }}</span>
            <span>{{ neighborOffsetLabel }}</span>
            <span>原始影格 {{ currentSample.metadata.frame_index }}</span>
          </div>
        </div>
        <div class="image-wrap" :class="{ loading: imagePending }">
          <img
            :key="currentSample.sample_id"
            :src="imageUrl(currentSample.image)"
            :alt="currentSample.sample_id"
            :class="{ visible: imageReady && !imageLoadError }"
            @load="handleImageLoaded"
            @error="handleImageError"
          />
          <div class="image-frame-chip">
            <strong>{{ keyFramePositionLabel }}</strong>
            <span>{{ neighborOffsetLabel }} · 原始影格 {{ currentSample.metadata.frame_index }}</span>
          </div>
          <div v-if="imagePending" class="image-loader" aria-live="polite">
            <span class="spinner"></span>
            <strong>正在載入影格</strong>
            <small>{{ currentSample.metadata.video_file }} · 關鍵影格 {{ currentSample.metadata.key_frame_index + 1 }}/5</small>
          </div>
          <div v-else-if="imageLoadError" class="image-loader error-state" aria-live="polite">
            <strong>影像載入失敗</strong>
            <small>{{ imageLoadError }}</small>
            <button type="button" @click="retryImage">重試</button>
          </div>
        </div>
        <div class="meta-grid">
          <span>技術：<strong>{{ skillLabel(currentSample.metadata.skill) }}</strong></span>
          <span>關鍵影格：<strong>{{ keyFrameLabel(currentSample.metadata.key_frame_name) }}</strong></span>
          <span>前後位移：<strong>{{ neighborOffsetLabel }}</strong></span>
          <span>原始影格編號：<strong>{{ currentSample.metadata.frame_index }}</strong></span>
          <span>慣用手：<strong>{{ handednessLabel(currentSample.metadata.handedness) }}</strong></span>
          <span>來源：<strong>{{ sourceLabel(currentSample.metadata.source_dataset) }}</strong></span>
        </div>
        <details class="angles">
          <summary>角度資料</summary>
          <div>
            <span v-for="(value, name) in currentSample.metadata.angles" :key="name">
              {{ angleLabel(String(name)) }}：<strong>{{ Number(value).toFixed(1) }}</strong>
            </span>
          </div>
        </details>
      </section>

      <form class="annotation" @submit.prevent="saveCurrent">
        <h2>專家標註</h2>
        <label>
          分數
          <input v-model.number="form.score" type="number" min="0" max="10" step="0.5" placeholder="0 到 10 分" />
        </label>
        <label>
          技術回饋
          <textarea v-model.trim="form.feedback" rows="5" placeholder="這個影格中動作正確或需要修正的地方"></textarea>
        </label>
        <label>
          修正建議
          <textarea v-model.trim="form.correction_suggestion" rows="4" placeholder="給選手的具體修正提示"></textarea>
        </label>
        <label>
          是否用於訓練
          <select v-model="form.usable_for_training">
            <option value="yes">是</option>
            <option value="no">否</option>
          </select>
        </label>
        <label>
          備註
          <textarea v-model.trim="form.notes" rows="3" placeholder="遮擋、骨架追蹤問題或判斷不明確之處"></textarea>
        </label>

        <div class="form-actions">
          <button type="button" @click="previousSample" :disabled="!canGoPrevious">上一張</button>
          <button type="submit" class="primary" :disabled="saving">{{ saving ? '儲存中...' : '儲存並下一張' }}</button>
          <button type="button" @click="nextSample" :disabled="!canGoNext">跳過</button>
        </div>
        <p v-if="saveStatus" class="save-status">{{ saveStatus }}</p>
      </form>
    </section>
  </main>
</template>

<script setup lang="ts">
import type { AnnotationSample, SavedAnnotation } from '~/types/annotation'

const PAGE_LIMIT = 200
const STATUS_BATCH_SIZE = 1000
type AnnotationState = 'unannotated' | 'in_progress' | 'annotated'

const config = useRuntimeConfig()
const annotationImageBaseUrl = computed(() => {
  const value = String(config.public.annotationImageBaseUrl || '').trim()
  return value.replace(/\/$/, '')
})
const annotator = useCookie('badminton_annotator', { default: () => '' })
const offset = ref(0)
const currentIndex = ref(0)
const allSamples = ref<AnnotationSample[]>([])
const filteredCount = ref(0)
const samples = ref<AnnotationSample[]>([])
const pending = ref(false)
const saving = ref(false)
const statusPending = ref(false)
const imagePending = ref(false)
const imageReady = ref(false)
const imageLoadError = ref('')
const imageRetryToken = ref(0)
const loadError = ref('')
const saveStatus = ref('')
const filters = reactive({
  skill: '',
  keyFrame: '',
  cohort: '',
  sourceDataset: '',
  annotationStatus: '' as '' | AnnotationState
})
const annotationStatusById = ref<Record<string, AnnotationState>>({})

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
const hasNextPage = computed(() => offset.value + PAGE_LIMIT < filteredCount.value)
const canGoPrevious = computed(() => currentIndex.value > 0 || offset.value > 0)
const canGoNext = computed(() => currentIndex.value < samples.value.length - 1 || hasNextPage.value)
const absolutePosition = computed(() => (filteredCount.value ? offset.value + currentIndex.value + 1 : 0))
const progressPercent = computed(() => {
  if (!filteredCount.value) return 0
  return Math.min(100, Math.max(0, (absolutePosition.value / filteredCount.value) * 100))
})
const pageStatus = computed(() => {
  if (!filteredCount.value) return '0 筆樣本'
  const start = offset.value + 1
  const end = offset.value + samples.value.length
  return `第 ${start}-${end} 筆，共 ${filteredCount.value} 筆`
})
const keyFramePositionLabel = computed(() => {
  if (!currentSample.value) return '沒有影格'
  return `關鍵影格 ${currentSample.value.metadata.key_frame_index + 1}/5`
})
const neighborOffsetLabel = computed(() => {
  if (!currentSample.value) return ''
  const offsetValue = currentSample.value.metadata.neighbor_offset
  if (offsetValue === 0) return '偵測到的關鍵影格'
  return offsetValue > 0 ? `關鍵影格後 ${offsetValue} 格` : `關鍵影格前 ${Math.abs(offsetValue)} 格`
})
const frameTitle = computed(() => {
  if (!currentSample.value) return ''
  return `${keyFrameLabel(currentSample.value.metadata.key_frame_name)} · ${neighborOffsetLabel.value}`
})
const frameSubtitle = computed(() => {
  if (!currentSample.value) return ''
  const meta = currentSample.value.metadata
  return `${skillLabel(meta.skill)} · ${handednessLabel(meta.handedness)} · ${meta.video_file}`
})
const statusText = computed(() => {
  const total = facets.value?.total ? `共 ${facets.value.total} 筆樣本` : '等待標註清單'
  return `${total} · 標註結果會透過伺服器寫入 Firestore`
})
const loadingText = computed(() => {
  if (statusPending.value) return '正在檢查標註狀態...'
  return '正在載入樣本...'
})

function skillLabel(skill: string) {
  const labels: Record<string, string> = {
    serve: '發球',
    clear: '高遠球',
    smash: '殺球',
    lift: '挑球'
  }
  return labels[skill] || skill
}

function keyFrameLabel(keyFrame: string) {
  const labels: Record<string, string> = {
    key_frame_0_start: '動作起始',
    key_frame_1_mid_start_peak: '起始到擊球點中段',
    key_frame_2_peak: '擊球點／最高點',
    key_frame_3_mid_peak_end: '擊球點到結束中段',
    key_frame_4_end: '動作結束'
  }
  return labels[keyFrame] || keyFrame
}

function cohortLabel(cohort: string) {
  const labels: Record<string, string> = {
    expert: '專家',
    beginner: '初學者'
  }
  return labels[cohort] || cohort
}

function sourceLabel(source: string) {
  const labels: Record<string, string> = {
    scoring_videos: '評分影片',
    'training_videos/nstc': 'NSTC 專家影片'
  }
  return labels[source] || source
}

function handednessLabel(handedness: string) {
  const labels: Record<string, string> = {
    left: '左手',
    right: '右手'
  }
  return labels[handedness] || handedness
}

function angleLabel(angleName: string) {
  const labels: Record<string, string> = {
    'Left Crotch Angle': '左髖角度',
    'Right Crotch Angle': '右髖角度',
    'Left Elbow Angle': '左手肘角度',
    'Right Elbow Angle': '右手肘角度',
    'Left Knee Angle': '左膝角度',
    'Right Knee Angle': '右膝角度',
    'Left Shoulder Angle': '左肩角度',
    'Right Shoulder Angle': '右肩角度',
    'Nose Left Shoulder Elbow Angle': '鼻子-左肩-左肘角度',
    'Nose Right Shoulder Elbow Angle': '鼻子-右肩-右肘角度'
  }
  return labels[angleName] || angleName
}

function imageUrl(path: string) {
  const baseUrl = staticImageUrl(path)
  return imageRetryToken.value ? `${baseUrl}?retry=${imageRetryToken.value}` : baseUrl
}

function staticImageUrl(path: string) {
  const normalizedPath = path.replace(/^\//, '')
  return annotationImageBaseUrl.value
    ? `${annotationImageBaseUrl.value}/${normalizedPath}`
    : `/annotation-images/${normalizedPath}`
}

function startImageLoad() {
  imagePending.value = true
  imageReady.value = false
  imageLoadError.value = ''
}

function handleImageLoaded() {
  imagePending.value = false
  imageReady.value = true
  imageLoadError.value = ''
  preloadNeighborImages()
}

function handleImageError() {
  imagePending.value = false
  imageReady.value = false
  imageLoadError.value = '圖片載入失敗，請確認標註影像已完成上傳或本機資料夾已準備完成。'
}

function retryImage() {
  imageRetryToken.value += 1
  startImageLoad()
}

function preloadNeighborImages() {
  if (!import.meta.client) return
  for (const index of [currentIndex.value - 1, currentIndex.value + 1]) {
    const sample = samples.value[index]
    if (!sample) continue
    const image = new Image()
    image.src = staticImageUrl(sample.image)
  }
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
    annotationStatusById.value = {
      ...annotationStatusById.value,
      [sample.sample_id]: annotationStateFromSaved(response.annotation)
    }
    const saved = response.annotation
    if (!saved) return
    form.score = saved.score
    form.feedback = saved.feedback || ''
    form.correction_suggestion = saved.correction_suggestion || ''
    form.usable_for_training = saved.usable_for_training || 'yes'
    form.notes = saved.notes || ''
    annotator.value = saved.annotator || annotator.value
  } catch (error) {
    saveStatus.value = error instanceof Error ? error.message : '無法載入已儲存的標註。'
  }
}

async function reload() {
  pending.value = true
  loadError.value = ''
  saveStatus.value = ''
  offset.value = 0
  currentIndex.value = 0
  try {
    await setCurrentPage(0)
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '無法載入樣本。'
  } finally {
    pending.value = false
  }
}

async function setCurrentPage(nextOffset: number, nextIndex = 0) {
  const filtered = await filteredSamples()
  filteredCount.value = filtered.length
  offset.value = Math.max(0, Math.min(nextOffset, Math.max(0, filtered.length - 1)))
  offset.value = Math.floor(offset.value / PAGE_LIMIT) * PAGE_LIMIT
  samples.value = filtered.slice(offset.value, offset.value + PAGE_LIMIT)
  currentIndex.value = Math.max(0, Math.min(nextIndex, Math.max(0, samples.value.length - 1)))

  if (currentSample.value) {
    imageRetryToken.value = 0
    startImageLoad()
    await hydrateForm(currentSample.value)
  } else {
    resetForm()
  }
}

async function nextPage() {
  if (!hasNextPage.value) return
  pending.value = true
  saveStatus.value = ''
  try {
    await setCurrentPage(offset.value + PAGE_LIMIT)
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '無法載入下一頁。'
  } finally {
    pending.value = false
  }
}

async function previousPage() {
  if (offset.value === 0) return
  pending.value = true
  saveStatus.value = ''
  try {
    await setCurrentPage(Math.max(0, offset.value - PAGE_LIMIT))
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '無法載入上一頁。'
  } finally {
    pending.value = false
  }
}

function baseFilteredSamples() {
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

async function filteredSamples() {
  const baseSamples = baseFilteredSamples()
  if (!filters.annotationStatus) return baseSamples

  await ensureAnnotationStatuses(baseSamples.map((sample) => sample.sample_id))
  return baseSamples.filter((sample) => {
    return annotationStatusById.value[sample.sample_id] === filters.annotationStatus
  })
}

function isAnnotated(sampleId: string) {
  return annotationStatusById.value[sampleId] === 'annotated'
}

function annotationStateFromSaved(saved: SavedAnnotation | null): AnnotationState {
  if (!saved) return 'unannotated'
  const hasScore = typeof saved.score === 'number'
  const hasFeedback = Boolean(saved.feedback?.trim())
  const hasCorrection = Boolean(saved.correction_suggestion?.trim())
  return hasScore && (hasFeedback || hasCorrection) ? 'annotated' : 'in_progress'
}

function annotationBadgeLabel(sampleId: string) {
  const status = annotationStatusById.value[sampleId]
  if (status === 'annotated') return '已完成'
  if (status === 'in_progress') return '標註中'
  if (status === 'unannotated') return '尚未標註'
  return '未檢查'
}

function annotationBadgeClass(sampleId: string) {
  const status = annotationStatusById.value[sampleId]
  if (status === 'annotated') return 'done'
  if (status === 'in_progress') return 'progress'
  if (status === 'unannotated') return 'todo'
  return 'unknown'
}

async function ensureAnnotationStatuses(sampleIds: string[]) {
  const missingIds = sampleIds.filter((sampleId) => annotationStatusById.value[sampleId] === undefined)
  if (!missingIds.length) return

  statusPending.value = true
  try {
    const nextStatuses = { ...annotationStatusById.value }
    for (let start = 0; start < missingIds.length; start += STATUS_BATCH_SIZE) {
      const batch = missingIds.slice(start, start + STATUS_BATCH_SIZE)
      const response = await $fetch<{ statuses: Record<string, AnnotationState> }>('/api/annotations/status', {
        method: 'POST',
        body: { sample_ids: batch }
      })
      Object.assign(nextStatuses, response.statuses)
    }
    annotationStatusById.value = nextStatuses
  } finally {
    statusPending.value = false
  }
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
        : '無法載入 /annotation_template.jsonl。部署前請先執行 prepare:assets。'
  } finally {
    pending.value = false
  }
}

async function selectSample(index: number) {
  currentIndex.value = index
  saveStatus.value = ''
  imageRetryToken.value = 0
  startImageLoad()
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
    const savedState = annotationStateFromSaved({
      sample_id: currentSample.value.sample_id,
      ...payload,
      metadata: currentSample.value.metadata
    })
    annotationStatusById.value = {
      ...annotationStatusById.value,
      [currentSample.value.sample_id]: savedState
    }
    saveStatus.value = '已儲存'
    if (filters.annotationStatus && filters.annotationStatus !== savedState) {
      await setCurrentPage(offset.value, currentIndex.value)
      saveStatus.value = '已儲存'
      return
    }
    nextSample()
  } catch (error) {
    saveStatus.value = error instanceof Error ? error.message : '儲存失敗。'
  } finally {
    saving.value = false
  }
}

function nextSample() {
  if (currentIndex.value < samples.value.length - 1) {
    void selectSample(currentIndex.value + 1)
  } else if (hasNextPage.value) {
    void nextPage()
  }
}

function previousSample() {
  if (currentIndex.value > 0) {
    void selectSample(currentIndex.value - 1)
  } else if (offset.value > 0) {
    const previousPageSize = Math.min(PAGE_LIMIT, offset.value)
    void setCurrentPage(Math.max(0, offset.value - PAGE_LIMIT), previousPageSize - 1)
  }
}

onMounted(async () => {
  await loadStaticManifest()
  await reload()
  if (currentSample.value) startImageLoad()
})
</script>

<style scoped>
.page {
  min-height: 100vh;
  background: #f4f6f8;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: rgb(255 255 255 / 0.94);
  backdrop-filter: blur(12px);
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.05);
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 21px;
  line-height: 1.15;
  letter-spacing: 0;
}

.topbar p,
.sample-list small,
.meta-grid span {
  color: var(--muted);
}

.top-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.annotator {
  width: 220px;
}

.filters {
  display: grid;
  grid-template-columns: repeat(5, minmax(140px, 1fr));
  gap: 12px;
  padding: 14px 20px;
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
  grid-template-columns: 280px minmax(0, 1fr) 390px;
  grid-template-areas:
    "progress progress progress"
    "list viewer annotation";
  gap: 16px;
  padding: 16px;
}

.progress-card {
  grid-area: progress;
  display: grid;
  grid-template-columns: auto minmax(140px, 1fr);
  align-items: center;
  gap: 16px;
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.05);
}

.progress-card span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.progress-card strong {
  display: block;
  margin-top: 2px;
  font-size: 18px;
}

.progress-track {
  height: 9px;
  overflow: hidden;
  border-radius: 999px;
  background: #e5e7eb;
}

.progress-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--accent);
  transition: width 180ms ease;
}

.sample-list {
  grid-area: list;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  align-content: start;
  max-height: calc(100vh - 184px);
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.05);
}

.sample-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 12px;
  border-bottom: 1px solid var(--line);
  background: #f8fafc;
}

.sample-list-header div {
  display: flex;
  gap: 6px;
}

.sample-list-header button {
  padding: 7px 10px;
}

.sample-list-scroll {
  overflow: auto;
}

.sample-list button {
  display: grid;
  grid-template-columns: 36px minmax(0, 1fr) auto;
  gap: 2px 8px;
  min-height: 64px;
  width: 100%;
  border: 0;
  border-bottom: 1px solid var(--line);
  border-radius: 0;
  text-align: left;
  background: #fff;
}

.sample-list button:hover {
  background: #f8fafc;
}

.sample-list button > span {
  align-self: center;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

.sample-list button strong {
  overflow: hidden;
  align-self: end;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sample-list small {
  grid-column: 2;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sample-list em {
  grid-column: 3;
  grid-row: 1 / span 2;
  align-self: center;
  border-radius: 999px;
  padding: 4px 8px;
  font-size: 11px;
  font-style: normal;
  font-weight: 800;
}

.sample-list em.done {
  background: #dcfce7;
  color: #166534;
}

.sample-list em.todo {
  background: #eef2ff;
  color: #3730a3;
}

.sample-list em.progress {
  background: #fef3c7;
  color: #92400e;
}

.sample-list em.unknown {
  background: #f2f4f7;
  color: #475467;
}

.sample-list .active {
  background: var(--accent-soft);
  box-shadow: inset 3px 0 0 var(--accent);
}

.viewer,
.annotation,
.message {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.05);
}

.viewer {
  grid-area: viewer;
  min-width: 0;
  padding: 12px;
}

.frame-context {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
}

.frame-context span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
}

.frame-context strong {
  display: block;
  margin-top: 3px;
  font-size: 18px;
}

.frame-context small {
  display: block;
  margin-top: 3px;
  color: var(--muted);
}

.frame-pills {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.frame-pills span {
  border: 1px solid #b7c8c4;
  border-radius: 999px;
  padding: 6px 9px;
  background: #fff;
  color: #0f766e;
  text-transform: none;
}

.image-wrap {
  position: relative;
  display: grid;
  place-items: center;
  min-height: 520px;
  background: #111827;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: inset 0 0 0 1px rgb(255 255 255 / 0.08);
}

.image-wrap img {
  max-width: 100%;
  max-height: 76vh;
  object-fit: contain;
  opacity: 0;
  transition: opacity 160ms ease;
}

.image-wrap img.visible {
  opacity: 1;
}

.image-wrap.loading {
  background:
    linear-gradient(90deg, rgb(17 24 39 / 0), rgb(255 255 255 / 0.07), rgb(17 24 39 / 0)),
    #111827;
  background-size: 220px 100%, 100% 100%;
  animation: shimmer 1.15s linear infinite;
}

.image-frame-chip {
  position: absolute;
  top: 12px;
  left: 12px;
  display: grid;
  gap: 2px;
  max-width: min(420px, calc(100% - 24px));
  border: 1px solid rgb(255 255 255 / 0.18);
  border-radius: 8px;
  padding: 9px 11px;
  color: #fff;
  background: rgb(15 23 42 / 0.82);
  box-shadow: 0 10px 24px rgb(0 0 0 / 0.22);
}

.image-frame-chip strong {
  font-size: 14px;
}

.image-frame-chip span {
  color: rgb(255 255 255 / 0.78);
  font-size: 12px;
}

.image-loader {
  position: absolute;
  inset: 0;
  display: grid;
  place-content: center;
  justify-items: center;
  gap: 10px;
  padding: 24px;
  color: #fff;
  text-align: center;
  background: rgb(17 24 39 / 0.58);
}

.image-loader small {
  max-width: 360px;
  color: rgb(255 255 255 / 0.76);
}

.image-loader button {
  background: #fff;
  color: #111827;
}

.spinner {
  width: 34px;
  height: 34px;
  border: 3px solid rgb(255 255 255 / 0.28);
  border-top-color: #fff;
  border-radius: 999px;
  animation: spin 720ms linear infinite;
}

.error-state {
  background: rgb(127 29 29 / 0.72);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes shimmer {
  from {
    background-position: -220px 0, 0 0;
  }

  to {
    background-position: calc(100% + 220px) 0, 0 0;
  }
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 12px 0 0;
  font-size: 13px;
}

.meta-grid span {
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
  background: #f8fafc;
}

.meta-grid strong {
  color: var(--text);
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
  grid-area: annotation;
  display: grid;
  align-content: start;
  gap: 14px;
  padding: 16px;
}

.annotation h2 {
  font-size: 18px;
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
  display: grid;
  justify-items: center;
  gap: 10px;
  margin: 20px;
  padding: 20px;
}

.loading-message .spinner {
  border-color: rgb(15 118 110 / 0.2);
  border-top-color: var(--accent);
}

.error {
  color: var(--danger);
}

@media (max-width: 1100px) {
  .workspace {
    grid-template-columns: 1fr;
    grid-template-areas:
      "progress"
      "viewer"
      "annotation"
      "list";
  }

  .sample-list {
    max-height: 220px;
  }

  .filters {
    grid-template-columns: repeat(2, 1fr);
  }

  .progress-card,
  .meta-grid {
    grid-template-columns: 1fr;
  }
}
</style>
