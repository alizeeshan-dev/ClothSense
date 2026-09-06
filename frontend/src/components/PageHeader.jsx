export default function PageHeader({ eyebrow, title, description, aside }) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {aside && <div className="page-header-aside">{aside}</div>}
    </header>
  )
}
