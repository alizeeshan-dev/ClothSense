export default function ConformalSet({ title, items, accent = false }) {
  return (
    <article className={`conformal-card${accent ? ' accent' : ''}`}>
      <h4>{title}</h4>
      <div className="chip-list">
        {items.length ? (
          items.map((item) => <span className="class-chip" key={item.id}>{item.name}</span>)
        ) : (
          <span className="empty-chip">Empty set</span>
        )}
      </div>
    </article>
  )
}
