import { cpSync, existsSync, mkdirSync, rmSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const publicDir = join(root, 'public')
const clientDir = join(root, '.nuxt', 'dist', 'client')
const manifestMetaDir = join(root, '.nuxt', 'manifest', 'meta')
const outputDir = join(root, '.output', 'public')

const requiredAssets = [
  [clientDir, outputDir],
  'annotation_schema.json',
  'annotation_template.csv',
  'annotation_template.jsonl'
]

if (process.env.SKIP_ANNOTATION_IMAGE_COPY === '1') {
  rmSync(join(outputDir, 'annotation-images'), { recursive: true, force: true })
} else {
  requiredAssets.push('annotation-images')
}

mkdirSync(outputDir, { recursive: true })

// Copy build metadata needed by Nuxt at runtime
if (existsSync(manifestMetaDir)) {
  const metaTarget = join(outputDir, '_nuxt', 'builds', 'meta')
  mkdirSync(metaTarget, { recursive: true })
  cpSync(manifestMetaDir, metaTarget, { recursive: true, force: true })
}

for (const asset of requiredAssets) {
  const source = Array.isArray(asset) ? asset[0] : join(publicDir, asset)
  const target = Array.isArray(asset) ? asset[1] : join(outputDir, asset)

  if (!existsSync(source)) {
    throw new Error(`Missing required public asset: ${source}`)
  }

  mkdirSync(statSync(source).isDirectory() ? target : dirname(target), { recursive: true })
  cpSync(source, target, { recursive: true, force: true })
}
