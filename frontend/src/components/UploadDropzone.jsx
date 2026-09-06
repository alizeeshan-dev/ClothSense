import { ImagePlus, UploadCloud } from 'lucide-react'
import { useRef, useState } from 'react'

export default function UploadDropzone({ file, note, onSelect, onError, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function choose(files) {
    if (files.length !== 1) {
      onError('Choose exactly one JPG, JPEG, or PNG image.')
      return
    }
    onSelect(files[0])
  }

  return (
    <div
      className={`dropzone${dragging ? ' dragging' : ''}${file ? ' has-file' : ''}`}
      onDragEnter={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        if (!disabled) choose(event.dataTransfer.files)
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        disabled={disabled}
        onChange={(event) => choose(event.target.files)}
      />
      <div className="dropzone-icon" aria-hidden="true">
        {file ? <ImagePlus size={26} /> : <UploadCloud size={26} />}
      </div>
      <div>
        <strong>{file ? file.name : 'Drop one garment image here'}</strong>
        <p>{file ? 'Ready to process in memory' : 'or choose a file from your computer'}</p>
      </div>
      <button type="button" className="secondary-button" onClick={() => inputRef.current?.click()} disabled={disabled}>
        {file ? 'Replace image' : 'Choose image'}
      </button>
      <small>{note}</small>
    </div>
  )
}
