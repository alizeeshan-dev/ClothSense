import { BarChart3, Filter } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { getResearchCharts, getResultsSummary } from '../api/clothsenseApi'
import MetricCard from '../components/MetricCard'
import PageHeader from '../components/PageHeader'
import ResearchChartCard from '../components/ResearchChartCard'
import { ErrorPanel, LoadingPanel } from '../components/StatePanel'
import { chartInterpretations, chartTopics } from '../data/chartCopy'
import { percent, readableName } from '../utils/formatters'

const topics = [
  ['all', 'All figures'],
  ['calibration', 'Calibration'],
  ['shift', 'Distribution shift'],
  ['conformal', 'Conformal'],
  ['selective', 'Abstention'],
  ['model', 'Model baseline'],
]

export default function ExperimentsPage() {
  const [summary, setSummary] = useState(null)
  const [charts, setCharts] = useState([])
  const [error, setError] = useState('')
  const [topic, setTopic] = useState('all')
  const [shiftName, setShiftName] = useState('gaussian_noise')
  const [requestKey, setRequestKey] = useState(0)

  useEffect(() => {
    let active = true
    setError('')
    Promise.all([getResultsSummary(), getResearchCharts()])
      .then(([summaryData, chartData]) => {
        if (!active) return
        setSummary(summaryData)
        setCharts(chartData.charts)
      })
      .catch(() => active && setError('The persisted experiment summary or chart index could not be loaded.'))
    return () => { active = false }
  }, [requestKey])

  const selectedShift = summary?.strongest_shifts.find((shift) => shift.shift_name === shiftName)
  const visibleCharts = useMemo(
    () => topic === 'all' ? charts : charts.filter((chart) => chartTopics[chart.name] === topic),
    [charts, topic],
  )

  return (
    <div className="page experiments-page">
      <PageHeader
        eyebrow="Three seeds · saved results only"
        title="Reliability changes when the data changes."
        description="Explore persisted clean and shifted evaluations. Filters select existing summaries and figures; they never launch training, inference, or recalibration."
        aside={<span className="research-pill"><BarChart3 size={16} /> 45 scenarios</span>}
      />

      {error && <ErrorPanel message={error} onRetry={() => setRequestKey((value) => value + 1)} />}
      {!summary && !error && <LoadingPanel />}

      {summary && (
        <>
          <section className="metric-grid experiment-metrics">
            <MetricCard label="Clean accuracy" value={percent(summary.clean.accuracy.mean)} detail="three-seed mean" />
            <MetricCard label="Raw ECE" value={percent(summary.clean.raw_ece.mean, 2)} detail="before calibration" tone="sand" />
            <MetricCard label="Calibrated ECE" value={percent(summary.clean.calibrated_ece.mean, 2)} detail="temperature-scaled" tone="olive" />
            <MetricCard label="Clean conformal coverage" value={percent(summary.clean.standard_conformal_coverage.mean)} detail={`${percent(1 - summary.demo_configuration.alpha, 0)} nominal`} />
            <MetricCard label="Demo acceptance" value={percent(summary.demo_configuration.clean_acceptance_rate.mean)} detail="hybrid policy" tone="charcoal" />
            <MetricCard label="Selective risk" value={percent(summary.demo_configuration.clean_selective_risk.mean)} detail="accepted predictions" tone="olive" />
          </section>

          <section className="shift-inspector card-surface">
            <div className="section-heading compact">
              <div><p className="eyebrow">Strongest-shift snapshot</p><h2>Compare one saved scenario</h2></div>
              <label className="select-control inline-select"><span><Filter size={15} /> Shift family</span><select value={shiftName} onChange={(event) => setShiftName(event.target.value)}>{summary.strongest_shifts.map((shift) => <option key={shift.shift_name} value={shift.shift_name}>{readableName(shift.shift_name)}</option>)}</select></label>
            </div>
            {selectedShift && (
              <div className="shift-stat-row">
                <div><span>Severity</span><strong>{readableName(selectedShift.severity_name)}</strong></div>
                <div><span>Accuracy</span><strong>{percent(selectedShift.accuracy.mean)}</strong><small>clean {percent(summary.clean.accuracy.mean)}</small></div>
                <div><span>Calibrated ECE</span><strong>{percent(selectedShift.calibrated_ece.mean, 2)}</strong><small>clean {percent(summary.clean.calibrated_ece.mean, 2)}</small></div>
                <div><span>Conformal coverage</span><strong>{percent(selectedShift.standard_conformal_coverage.mean)}</strong><small>target {percent(1 - summary.demo_configuration.alpha, 0)}</small></div>
              </div>
            )}
          </section>

          <section className="interpretation-strip">
            <div><strong>Calibration</strong><span>Temperature scaling improves confidence quality on familiar data.</span></div>
            <div><strong>Coverage</strong><span>Nominal conformal coverage can weaken after distribution shift.</span></div>
            <div><strong>Abstention</strong><span>Lower selective risk comes at the cost of accepting fewer predictions.</span></div>
          </section>

          <section className="section-block">
            <div className="section-heading chart-toolbar">
              <div><p className="eyebrow">Research figures</p><h2>Generated from persisted CSV results</h2></div>
              <div className="filter-pills" role="group" aria-label="Filter research figures by topic">
                {topics.map(([value, label]) => <button type="button" key={value} className={topic === value ? 'active' : ''} onClick={() => setTopic(value)}>{label}</button>)}
              </div>
            </div>
            {visibleCharts.length ? (
              <div className="chart-grid">
                {visibleCharts.map((chart) => <ResearchChartCard key={chart.name} chart={chart} interpretation={chartInterpretations[chart.name]} featured={chart.name.includes('coverage_vs_shift')} />)}
              </div>
            ) : <div className="empty-state">No persisted figures match this topic.</div>}
          </section>
        </>
      )}
    </div>
  )
}
