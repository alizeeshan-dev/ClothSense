import { Check, CircleHelp } from 'lucide-react'
import ConformalSet from './ConformalSet'
import ProbabilityBars from './ProbabilityBars'
import { percent, readableName } from '../utils/formatters'

export default function PredictionResult({ result }) {
  const accepted = result.abstention.accepted
  const nominal = (1 - result.alpha) * 100
  return (
    <section className="results-stack" aria-live="polite">
      <article className={`decision-card ${accepted ? 'accepted' : 'uncertain'}`}>
        <div className="decision-icon" aria-hidden="true">
          {accepted ? <Check size={25} /> : <CircleHelp size={25} />}
        </div>
        <div className="decision-primary">
          <p className="eyebrow">Model prediction</p>
          <h2>{result.predicted_class.name}</h2>
          <span className="decision-label">{accepted ? 'Accepted' : 'Uncertain'}</span>
        </div>
        <div className="confidence-pair">
          <div><span>Raw confidence</span><strong>{percent(result.raw_confidence)}</strong></div>
          <div><span>Calibrated</span><strong>{percent(result.calibrated_confidence)}</strong></div>
        </div>
        <p className="decision-reason">{result.decision_reason}</p>
      </article>

      <ProbabilityBars classes={result.top_classes} />

      <section className="conformal-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Set-valued uncertainty</p>
            <h3>Conformal prediction sets</h3>
          </div>
          <span className="alpha-badge">α {result.alpha} · nominal {nominal.toFixed(0)}%</span>
        </div>
        <div className="conformal-grid">
          <ConformalSet title="Standard conformal" items={result.standard_conformal_set} />
          <ConformalSet title="Class-conditional" items={result.class_conditional_conformal_set} accent />
        </div>
        <p className="micro-copy">A larger set means the model considers several classes plausible. Coverage targets need not hold after distribution shift.</p>
      </section>

      <div className="method-footnote">
        Policy: <strong>{readableName(result.abstention.policy)}</strong>
        {result.abstention.threshold !== null && <> · threshold <strong>{result.abstention.threshold}</strong></>}
        {' · '}temperature <strong>{result.temperature.toFixed(3)}</strong>
      </div>
    </section>
  )
}
