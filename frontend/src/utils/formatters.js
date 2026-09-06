export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function decimal(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(digits)
}

export function readableName(value = '') {
  return value
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function fileSize(bytes) {
  if (!bytes) return null
  return bytes >= 1_000_000
    ? `${(bytes / 1_000_000).toFixed(bytes % 1_000_000 ? 1 : 0)} MB`
    : `${Math.ceil(bytes / 1000)} KB`
}
