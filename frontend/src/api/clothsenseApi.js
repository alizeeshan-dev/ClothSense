const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

function endpoint(path) {
  return `${API_BASE}${path}`
}

async function readJson(response) {
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = payload?.detail
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(' ')
      : detail || 'The ClothSense backend could not complete this request.'
    throw new Error(message)
  }
  return payload
}

export async function getClasses() {
  const response = await fetch(endpoint('/api/classes'))
  const classes = await readJson(response)
  const alphaHeader = response.headers.get('X-Supported-Alphas')
  return {
    classes,
    maxUploadBytes: Number(response.headers.get('X-Max-Upload-Bytes')) || null,
    supportedAlphas: alphaHeader
      ? alphaHeader.split(',').map(Number).filter(Number.isFinite)
      : [],
    defaultInvert: response.headers.get('X-Default-Invert') === 'true',
  }
}

export async function classifyImage({ file, invert, alpha }) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('invert', String(invert))
  if (alpha !== undefined && alpha !== null && alpha !== '') {
    formData.append('alpha', String(alpha))
  }
  const response = await fetch(endpoint('/api/classify'), {
    method: 'POST',
    body: formData,
  })
  return readJson(response)
}

export async function getResultsSummary() {
  return readJson(await fetch(endpoint('/api/results/summary')))
}

export async function getResearchCharts() {
  return readJson(await fetch(endpoint('/api/results/charts')))
}

export function assetUrl(path) {
  return endpoint(path)
}
