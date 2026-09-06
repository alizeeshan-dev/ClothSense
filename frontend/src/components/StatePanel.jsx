import { AlertCircle, LoaderCircle } from 'lucide-react'

export function LoadingPanel({ label = 'Loading saved research results…' }) {
  return (
    <div className="state-panel" role="status">
      <LoaderCircle className="spin" size={20} aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function ErrorPanel({ message, onRetry }) {
  return (
    <div className="state-panel error-panel" role="alert">
      <AlertCircle size={20} aria-hidden="true" />
      <span>{message}</span>
      {onRetry && (
        <button className="text-button" type="button" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  )
}
