import { percent } from '../utils/formatters'

export default function ProbabilityBars({ classes = [] }) {
  return (
    <section className="probability-section">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Calibrated ranking</p>
          <h3>Top three classes</h3>
        </div>
      </div>
      <div className="probability-list">
        {classes.map((item, index) => (
          <div className="probability-row" key={item.id}>
            <span className="probability-rank">0{index + 1}</span>
            <span className="probability-name">{item.name}</span>
            <div className="probability-track" aria-hidden="true">
              <span style={{ width: `${Math.max(item.probability * 100, 1)}%` }} />
            </div>
            <strong>{percent(item.probability)}</strong>
          </div>
        ))}
      </div>
    </section>
  )
}
