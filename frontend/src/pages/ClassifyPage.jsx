import { FlaskConical, LoaderCircle } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { classifyImage, getClasses } from '../api/clothsenseApi'
import DomainWarning from '../components/DomainWarning'
import ImageComparison from '../components/ImageComparison'
import PageHeader from '../components/PageHeader'
import PredictionResult from '../components/PredictionResult'
import { ErrorPanel } from '../components/StatePanel'
import UploadDropzone from '../components/UploadDropzone'
import { fileSize } from '../utils/formatters'

const acceptedExtensions = ['jpg', 'jpeg', 'png']

export default function ClassifyPage() {
  const [metadata, setMetadata] = useState(null)
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState('')
  const [invert, setInvert] = useState(false)
  const [alpha, setAlpha] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getClasses()
      .then((value) => {
        setMetadata(value)
        setInvert(value.defaultInvert)
        setAlpha(String(value.supportedAlphas.includes(0.1) ? 0.1 : value.supportedAlphas[0] ?? ''))
      })
      .catch(() => setError('Could not load classifier settings. Start the local FastAPI server and try again.'))
  }, [])

  useEffect(() => () => { if (originalUrl) URL.revokeObjectURL(originalUrl) }, [originalUrl])

  const uploadNote = useMemo(() => {
    const limit = fileSize(metadata?.maxUploadBytes)
    return `JPG, JPEG or PNG · one image${limit ? ` · up to ${limit}` : ''}`
  }, [metadata])

  function selectFile(nextFile) {
    setError('')
    const extension = nextFile.name.split('.').pop()?.toLowerCase()
    if (!acceptedExtensions.includes(extension) || !['image/jpeg', 'image/png'].includes(nextFile.type)) {
      if (originalUrl) URL.revokeObjectURL(originalUrl)
      setFile(null)
      setOriginalUrl('')
      setResult(null)
      setError('Choose a valid JPG, JPEG, or PNG image.')
      return
    }
    if (metadata?.maxUploadBytes && nextFile.size > metadata.maxUploadBytes) {
      if (originalUrl) URL.revokeObjectURL(originalUrl)
      setFile(null)
      setOriginalUrl('')
      setResult(null)
      setError(`That image is larger than the ${fileSize(metadata.maxUploadBytes)} limit.`)
      return
    }
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setFile(nextFile)
    setOriginalUrl(URL.createObjectURL(nextFile))
    setResult(null)
  }

  async function submit(event) {
    event.preventDefault()
    if (!file) {
      setError('Choose one image before classifying.')
      return
    }
    setLoading(true)
    setError('')
    try {
      setResult(await classifyImage({ file, invert, alpha: Number(alpha) }))
    } catch (requestError) {
      setResult(null)
      setError(requestError.message || 'The image could not be classified.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page classify-page">
      <PageHeader
        eyebrow="Interactive inference"
        title="See what the model sees."
        description="Compare a photograph with its exact 28×28 model input, then inspect calibrated confidence, conformal sets, and the selected abstention decision."
        aside={<span className="research-pill"><FlaskConical size={16} /> Experimental classifier</span>}
      />

      <form className="classifier-workspace" onSubmit={submit}>
        <section className="upload-controls card-surface">
          <div className="section-heading compact"><div><p className="eyebrow">Step 1</p><h2>Choose one image</h2></div></div>
          <UploadDropzone file={file} note={uploadNote} onSelect={selectFile} onError={setError} disabled={loading} />
          <div className="control-row">
            <label className="toggle-control">
              <input type="checkbox" checked={invert} onChange={(event) => { setInvert(event.target.checked); setResult(null) }} disabled={loading} />
              <span className="toggle-track" aria-hidden="true"><span /></span>
              <span><strong>Invert intensities</strong><small>Useful for dark garments on bright backgrounds</small></span>
            </label>
            <label className="select-control">
              <span>Conformal alpha</span>
              <select value={alpha} onChange={(event) => { setAlpha(event.target.value); setResult(null) }} disabled={loading || !metadata?.supportedAlphas.length}>
                {metadata?.supportedAlphas.map((value) => <option key={value} value={value}>{value.toFixed(2)} · {(100 - value * 100).toFixed(0)}% nominal</option>)}
              </select>
            </label>
          </div>
          {error && <ErrorPanel message={error} />}
          <button className="primary-button classify-button" type="submit" disabled={!file || loading || !alpha}>
            {loading ? <><LoaderCircle className="spin" size={18} /> Classifying in memory…</> : 'Classify image'}
          </button>
        </section>

        <section className="visual-input card-surface">
          <div className="section-heading compact"><div><p className="eyebrow">Step 2</p><h2>Inspect the transformation</h2></div></div>
          <ImageComparison originalUrl={originalUrl} processedImage={result?.processed_image} />
        </section>
      </form>

      {result && <PredictionResult result={result} />}
      <DomainWarning message={result?.domain_limitation} />
    </div>
  )
}
