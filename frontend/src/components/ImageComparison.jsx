import { ScanSearch } from 'lucide-react'

export default function ImageComparison({ originalUrl, processedImage }) {
  return (
    <section className="image-comparison" aria-label="Original and processed image comparison">
      <article className="image-panel">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Source</p>
            <h3>Original image</h3>
          </div>
        </div>
        <div className="image-stage original-stage">
          {originalUrl ? <img src={originalUrl} alt="Selected original garment upload" /> : <span>No image selected</span>}
        </div>
      </article>
      <div className="transform-arrow" aria-hidden="true">→</div>
      <article className="image-panel model-panel">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">CNN input</p>
            <h3>Exact 28×28 model input</h3>
          </div>
          <ScanSearch size={20} aria-hidden="true" />
        </div>
        <div className="image-stage processed-stage">
          {processedImage ? (
            <img src={processedImage.data_url} alt="Exact 28 by 28 grayscale image supplied to the model" />
          ) : (
            <span>Appears after classification</span>
          )}
        </div>
        <p className="micro-copy">Pixels are enlarged without smoothing so the transformation stays visible.</p>
      </article>
    </section>
  )
}
