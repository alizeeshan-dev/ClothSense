import { assetUrl } from '../api/clothsenseApi'

export default function ResearchChartCard({ chart, interpretation, featured = false }) {
  return (
    <figure className={`chart-card${featured ? ' featured-chart' : ''}`}>
      <div className="chart-copy">
        <p className="eyebrow">Persisted research figure</p>
        <h3>{chart.title}</h3>
        {interpretation && <p>{interpretation}</p>}
      </div>
      <a href={assetUrl(chart.url)} target="_blank" rel="noreferrer" aria-label={`Open ${chart.title} full size`}>
        <img src={assetUrl(chart.url)} alt={chart.title} loading="lazy" />
      </a>
    </figure>
  )
}
