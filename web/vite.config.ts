import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 開発中は /api へのリクエストを FastAPI (uvicorn api:app --port 8000) に転送する
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
