import { ImageOff, Images, Microscope } from 'lucide-react'
import { useEffect, useState } from 'react'
import { assetUrl, getResultsSummary } from '../api/clothsenseApi'
import PageHeader from '../components/PageHeader'
import { ErrorPanel } from '../components/StatePanel'
import { percent } from '../utils/formatters'

const savedExamples = [
  {
    title: 'Fashion-MNIST reference samples',
    subtitle: 'Centered catalogue-style inputs',
    image: '/charts/fashion_mnist_samples.png',
    description: 'A fixed training-domain sample grid used to verify class balance and preprocessing.',
  },
  {
    title: 'Controlled shift verification',
    subtitle: 'Noise, rotation, blur, lighting, and imbalance',
    image: '/charts/shift_verification.png',
    description: 'Representative test images verify that configured corruptions operate in raw pixel space.',
  },
]

export default function GalleryPage() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getResultsSummary().then(setSummary).catch(() => setError('Saved failure-pattern summaries could not be loaded.'))
  }, [])

  const noise = summary?.strongest_shifts.find((item) => item.shift_name === 'gaussian_noise')
  const rotation = summary?.strongest_shifts.find((item) => item.shift_name === 'rotation')

  return (
    <div className="page gallery-page">
      <PageHeader
        eyebrow="Research examples"
        title="Inputs, transformations, and failure patterns."
        description="The gallery uses only existing project artifacts. Personal photographs will appear after the qualitative demo workflow is run."
        aside={<span className="research-pill"><Images size={16} /> Saved artifacts</span>}
      />

      <section className="gallery-grid">
        {savedExamples.map((example) => (
          <article className="example-card" key={example.title}>
            <div className="example-image"><img src={assetUrl(example.image)} alt={example.title} /></div>
            <div className="example-copy"><p className="eyebrow">{example.subtitle}</p><h2>{example.title}</h2><p>{example.description}</p></div>
          </article>
        ))}
      </section>

      <section className="section-block">
        <div className="section-heading"><div><p className="eyebrow">Observed failure patterns</p><h2>Where uncertainty becomes fragile</h2></div></div>
        {error && <ErrorPanel message={error} />}
        {summary && (
          <div className="failure-grid">
            <article><span>01</span><Microscope size={22} /><h3>Strong noise</h3><p>Accuracy falls to <strong>{percent(noise.accuracy.mean)}</strong> while calibrated ECE rises to <strong>{percent(noise.calibrated_ece.mean, 2)}</strong>.</p></article>
            <article><span>02</span><Microscope size={22} /><h3>Strong rotation</h3><p>Observed conformal coverage falls to <strong>{percent(rotation.standard_conformal_coverage.mean)}</strong> against a <strong>{percent(1 - summary.demo_configuration.alpha, 0)}</strong> nominal target.</p></article>
            <article><span>03</span><Microscope size={22} /><h3>Real-photo mismatch</h3><p>Backgrounds, pose, texture, and lighting differ from centered Fashion-MNIST images. Abstention describes uncertainty; it is not an OOD detector.</p></article>
          </div>
        )}
      </section>

      <section className="empty-gallery-state">
        <div className="empty-gallery-icon"><ImageOff size={28} /></div>
        <div><p className="eyebrow">Personal-photo study</p><h2>No personal examples have been added yet.</h2><p>Place personally sourced JPG/JPEG/PNG images in <code>data/personal_photos/</code>, optionally add expected labels, then run <code>python -m scripts.process_demo_images</code>. No example photographs are fabricated here.</p></div>
      </section>
    </div>
  )
}
