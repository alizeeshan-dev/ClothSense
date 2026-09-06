import { ArrowRight, CircleGauge, Layers3, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getResearchCharts, getResultsSummary } from '../api/clothsenseApi'
import MetricCard from '../components/MetricCard'
import ResearchChartCard from '../components/ResearchChartCard'
import { ErrorPanel, LoadingPanel } from '../components/StatePanel'
import { chartInterpretations } from '../data/chartCopy'
import { percent } from '../utils/formatters'

const pipeline = [
  'Fashion-MNIST',
  'Compact CNN',
  'Raw confidence',
  'Temperature calibration',
  'Conformal sets',
  'Abstention',
  'Shift evaluation',
]

export default function OverviewPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [requestKey, setRequestKey] = useState(0)

  useEffect(() => {
    let active = true
    setError('')
    Promise.all([getResultsSummary(), getResearchCharts()])
      .then(([summary, chartResponse]) => active && setData({ summary, charts: chartResponse.charts }))
      .catch(() => active && setError('Saved research results could not be loaded. Is the local API running?'))
    return () => { active = false }
  }, [requestKey])

  const reliability = data?.charts.find((chart) => chart.name.includes('reliability_raw'))
  const comparison = data?.charts.find((chart) => chart.name.includes('clean_vs_strongest'))

  return (
    <div className="page overview-page">
      <section className="overview-hero">
        <div className="hero-copy">
          <p className="eyebrow">Local uncertainty research · Fashion-MNIST</p>
          <h1>Confidence is useful.<br /><em>Context makes it trustworthy.</em></h1>
          <p>
            ClothSense studies whether machine-learning confidence remains trustworthy when clothing images differ from the data used to train the model.
          </p>
          <div className="hero-actions">
            <Link to="/classify" className="primary-button">Classify an image <ArrowRight size={17} /></Link>
            <Link to="/experiments" className="secondary-button">Explore results</Link>
          </div>
        </div>
        <div className="hero-orbit" aria-label="Research focus: prediction, calibration, and coverage">
          <div className="orbit-center"><span>CS</span><small>uncertainty<br />in context</small></div>
          <div className="orbit-node node-one"><CircleGauge size={21} /><span>Calibration</span></div>
          <div className="orbit-node node-two"><Layers3 size={21} /><span>Prediction sets</span></div>
          <div className="orbit-node node-three"><ShieldCheck size={21} /><span>Abstention</span></div>
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><p className="eyebrow">Research pipeline</p><h2>One model, several views of uncertainty</h2></div>
          <p>Each stage reuses saved model outputs. No method retrains the classifier on shifted data.</p>
        </div>
        <div className="pipeline" aria-label={pipeline.join(' then ')}>
          {pipeline.map((stage, index) => (
            <div className="pipeline-step" key={stage}>
              <span>{String(index + 1).padStart(2, '0')}</span><strong>{stage}</strong>
              {index < pipeline.length - 1 && <ArrowRight size={16} aria-hidden="true" />}
            </div>
          ))}
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading"><div><p className="eyebrow">Saved findings</p><h2>Clean performance, with uncertainty attached</h2></div></div>
        {error && <ErrorPanel message={error} onRetry={() => setRequestKey((value) => value + 1)} />}
        {!data && !error && <LoadingPanel />}
        {data && (
          <div className="metric-grid overview-metrics">
            <MetricCard label="Clean accuracy" value={percent(data.summary.clean.accuracy.mean)} detail="mean across 3 seeds" />
            <MetricCard label="Calibrated ECE" value={percent(data.summary.clean.calibrated_ece.mean, 2)} detail={`raw ${percent(data.summary.clean.raw_ece.mean, 2)}`} tone="olive" />
            <MetricCard label="Conformal coverage" value={percent(data.summary.clean.standard_conformal_coverage.mean)} detail={`nominal ${percent(1 - data.summary.demo_configuration.alpha, 0)}`} tone="sand" />
            <MetricCard label="Accepted by demo policy" value={percent(data.summary.demo_configuration.clean_acceptance_rate.mean)} detail={`${percent(data.summary.demo_configuration.clean_selective_risk.mean)} selective risk`} tone="charcoal" />
          </div>
        )}
      </section>

      {data && (
        <section className="section-block overview-charts">
          <div className="section-heading"><div><p className="eyebrow">At a glance</p><h2>Familiar data versus shifted data</h2></div><Link to="/experiments" className="text-link">View all figures <ArrowRight size={15} /></Link></div>
          <div className="chart-grid two-column">
            {reliability && <ResearchChartCard chart={reliability} interpretation={chartInterpretations[reliability.name]} />}
            {comparison && <ResearchChartCard chart={comparison} interpretation={chartInterpretations[comparison.name]} />}
          </div>
        </section>
      )}

      <section className="research-question-card">
        <div><p className="eyebrow">Central question</p><h2>What happens when the input distribution moves?</h2></div>
        <p>Calibration and conformal prediction communicate uncertainty well on familiar data, but their reliability and coverage can weaken as noise, rotation, blur, lighting, or class balance changes.</p>
      </section>
    </div>
  )
}
