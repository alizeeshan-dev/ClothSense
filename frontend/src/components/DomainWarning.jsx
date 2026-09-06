import { Info } from 'lucide-react'

export default function DomainWarning({ message }) {
  return (
    <aside className="domain-warning">
      <Info size={21} aria-hidden="true" />
      <div>
        <strong>Interpret in context</strong>
        <p>
          {message ||
            'Fashion-MNIST contains centered 28×28 grayscale garments. Ordinary photographs may differ substantially; uncertainty is not proof of out-of-distribution detection.'}
        </p>
      </div>
    </aside>
  )
}
