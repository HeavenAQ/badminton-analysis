const isNetlifyBuild = process.env.NETLIFY === 'true' || process.env.NITRO_PRESET === 'netlify'

export default defineNuxtConfig({
  compatibilityDate: '2025-01-01',
  devtools: {
    enabled: process.env.NUXT_DEVTOOLS === 'true'
  },
  css: ['~/assets/main.css'],
  runtimeConfig: {
    annotationRoot: process.env.ANNOTATION_ROOT || '../llm-annotations',
    gcpProjectId: process.env.GCP_PROJECT_ID || 'moe-linebot-2025',
    firestoreDatabaseId: process.env.FIRESTORE_DATABASE_ID || 'badminton-annotations',
    firestoreCollection: process.env.FIRESTORE_COLLECTION || 'badminton_vlm_annotations',
    public: {
      annotationImageBaseUrl: process.env.NUXT_PUBLIC_ANNOTATION_IMAGE_BASE_URL || ''
    }
  },
  modules: ['@nuxtjs/google-fonts'],
  googleFonts: {
    families: {
      Inter: [400, 500, 600, 700]
    }
  },
  nitro: {
    preset: isNetlifyBuild ? 'netlify' : 'node-server',
    externals: {
      inline: ['@vue/devtools-api', 'vue-router']
    },
    routeRules: {
      '/annotation-images/**': {
        headers: {
          'cache-control': 'public, max-age=3600'
        }
      }
    }
  }
})
