export type Angles = Record<string, number>

export interface AnnotationSample {
  sample_id: string
  image: string
  metadata: {
    source_dataset: string
    skill_zh: string
    skill: string
    handedness: string
    handedness_source: string
    cohort: string
    source_group: string
    video_file: string
    key_frame_index: number
    key_frame_name: string
    neighbor_offset: number
    frame_index: number
    angles: Angles
  }
  expert_annotation: {
    score: string | number
    feedback: string
    correction_suggestion: string
    usable_for_training: string
    notes: string
  }
}

export interface SavedAnnotation {
  sample_id: string
  score: number | null
  feedback: string
  correction_suggestion: string
  usable_for_training: 'yes' | 'no'
  annotator: string
  notes: string
  updated_at?: unknown
  metadata?: AnnotationSample['metadata']
}
