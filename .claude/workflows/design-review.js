export const meta = {
  name: 'design-review',
  description: 'Severity-gated design review: dimension reviewers -> adversarial verify -> findings at/above a threshold (converged when none).',
  whenToUse: 'Run a top-to-bottom design review that STOPS itself when nothing meets the severity bar. Pass args {threshold:"high"|"medium"|"low", maxFindings?, focus?}.',
  phases: [
    { title: 'Review', detail: 'one reviewer per architectural dimension' },
    { title: 'Verify', detail: 'adversarially verify each finding against the code' },
    { title: 'Synthesize', detail: 'rank the findings that meet the severity bar' },
  ],
}

// ---- args: {threshold, maxFindings, focus} (also accepts a bare threshold string) ----
const A = typeof args === 'string' ? { threshold: args } : (args || {})
const THRESHOLD = ['high', 'medium', 'low'].includes(A.threshold) ? A.threshold : 'medium'
const MAX_FINDINGS = Number.isInteger(A.maxFindings) ? A.maxFindings : 5
const FOCUS = typeof A.focus === 'string' && A.focus.trim() ? A.focus.trim() : ''
const SEV_RANK = { high: 0, medium: 1, low: 2 }
const THR_RANK = SEV_RANK[THRESHOLD]
const meetsBar = (sev) => (SEV_RANK[sev] ?? 3) <= THR_RANK

const ROOT = '/Users/fillmore/EarthSystem/DAVINCI'

const CONTEXT = `
You are reviewing the DAVINCI codebase (Data Analysis and Visual Intelligence for Climate), a type-safe Python
toolkit for evaluating atmospheric chemistry / air-quality datasets, based on MELODIES-MONET. Repo root: ${ROOT};
main package davinci_monet/.

Architecture: core/ (protocols, registry, base.py, exceptions), config/ (pydantic schema + parser), datasets/
(model readers + obs handlers), pairing/ (engine + geometry strategies point/track/profile/swath/grid +
intermediate_grid), plots/ (renderers, style.py, base.py, labels.py, contracts.py), stats/ (metrics, calculator,
output), pipeline/ (runner, stages/, display, reporting), ai/ (optional summary), cli/.

Conventions/goals: xarray-only data model; geometry-driven pairing; pairing roles x/y (diffs y-x); CESM surface is
lev=-1; module-size goal < 500 lines; mypy strict; plugin registry via decorators. Local gates (run in the 'davinci'
conda env) are the source of truth; GitHub Actions is disabled for the repo.

This is a DESIGN review, not a bug hunt: architecture, abstractions, type/contract design, module boundaries,
coupling/cohesion, duplication, leaky abstractions, inconsistent/competing conventions, weak invariants,
extensibility friction, maintainability risk. Read the ACTUAL current code with Read/Grep/Glob before asserting
anything; cite file:line for every claim. It is CORRECT and expected to report ZERO findings for a dimension if the
code is sound — do not invent issues to fill a quota.${FOCUS ? `\n\nEXTRA FOCUS for this run: ${FOCUS}` : ''}
`

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          dimension: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          locations: { type: 'array', items: { type: 'string' } },
          concern: { type: 'string' },
          evidence: { type: 'string' },
          impact: { type: 'string' },
          recommendation: { type: 'string' },
        },
        required: ['title', 'dimension', 'severity', 'locations', 'concern', 'evidence', 'impact', 'recommendation'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    isReal: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    severity_adjusted: { type: 'string', enum: ['high', 'medium', 'low'] },
    assessment: { type: 'string' },
    corrections: { type: 'string' },
  },
  required: ['isReal', 'confidence', 'severity_adjusted', 'assessment', 'corrections'],
}

const DIMENSIONS = [
  { key: 'architecture-boundaries', focus: 'Overall architecture, module boundaries, layering, coupling/cohesion, registry/protocol enforcement, god-modules, backwards/circular deps.' },
  { key: 'config-and-stage-contracts', focus: 'Config system + pipeline stage contracts: typed vs stringly-typed config access, data keys between stages, schema cohesion/size, validation vs runtime assumptions, invariants in types vs imperative checks.' },
  { key: 'pairing-engine', focus: 'pairing/engine.py + strategies/: dispatch by geometry, duplication across _create_paired_output, candidate-name probing, the x_/y_ vs source_label shim, untyped option kwargs, whether the strategy seam is right.' },
  { key: 'plotting-renderers', focus: 'plots/base.py + renderers/ + style.py + labels.py + contracts.py: renderer hierarchy & shared-logic factoring, duplication, single-source vs pairwise arity, 3D-track plotter overlap, coupling to paired-data naming.' },
  { key: 'data-model-types', focus: 'xarray-only contract, geometry shapes, core/base.py axis/canonical helpers, attrs-based metadata: invariant enforcement vs convention, robustness of attrs-as-typing, stringly-typed dispatch.' },
  { key: 'pipeline-observability', focus: 'runner.py + stages/ + display.py + reporting.py: stage contract & error surfacing, display/reporting over-build vs execution tangle, orchestration vs presentation separation.' },
  { key: 'consistency-duplication-size', focus: 'Cross-cutting consistency, duplication, dead code, < 500-line guideline (use wc -l), competing conventions and retained fallbacks, inconsistent error-raising, naming drift.' },
  { key: 'extensibility-seams', focus: 'Plugin model: how new datasets/strategies/plots/metrics are added (registry decorators), metric registry vs hardcoded switchboards, whether the registry is load-bearing or bypassed, discoverability of contracts.' },
]

phase('Review')
const reviewed = await pipeline(
  DIMENSIONS,
  d => agent(`${CONTEXT}\n\nDIMENSION: ${d.focus}\nReport design-level findings only (possibly none).`,
    { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  (review, d) => {
    const findings = (review && review.findings) || []
    if (!findings.length) return []
    return parallel(findings.map(f => () =>
      agent(`${CONTEXT}

ADVERSARIAL VERIFICATION. A reviewer reported this DESIGN finding. Skeptically verify it against the actual code —
open the cited files/lines and confirm or refute. Default to isReal=false if the evidence does not hold up, if it is
a subjective style nit dressed as a design flaw, if it misreads the code, or if it is already mitigated. Mark
isReal=true only for a genuine, accurately-described design issue with real impact. Set severity_adjusted to what the
evidence supports.

FINDING (dimension: ${d.key}):
title: ${f.title}
severity(claimed): ${f.severity}
locations: ${JSON.stringify(f.locations)}
concern: ${f.concern}
evidence: ${f.evidence}
impact: ${f.impact}
recommendation: ${f.recommendation}`,
        { label: `verify:${d.key}:${(f.title || '').slice(0, 28)}`, phase: 'Verify', schema: VERDICT_SCHEMA })
        .then(v => ({ ...f, dimension: d.key, verdict: v }))
        .catch(() => null)
    ))
  }
)

const confirmed = reviewed
  .flat()
  .filter(Boolean)
  .filter(f => f.verdict && f.verdict.isReal)
  .map(f => ({ ...f, sev: (f.verdict && f.verdict.severity_adjusted) || f.severity }))

const atBar = confirmed
  .filter(f => meetsBar(f.sev))
  .sort((a, b) => (SEV_RANK[a.sev] ?? 3) - (SEV_RANK[b.sev] ?? 3))

const maxSeverity = confirmed.length
  ? confirmed.reduce((m, f) => ((SEV_RANK[f.sev] ?? 3) < (SEV_RANK[m] ?? 3) ? f.sev : m), 'low')
  : null

log(`Confirmed ${confirmed.length} finding(s); ${atBar.length} at/above '${THRESHOLD}'. ` +
  (atBar.length ? '' : 'CONVERGED.'))

// Severity terminator: nothing meets the bar -> converged, no synthesis needed.
if (!atBar.length) {
  return { threshold: THRESHOLD, converged: true, maxSeverity, findings: [] }
}

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string' },
          locations: { type: 'array', items: { type: 'string' } },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          summary: { type: 'string', description: 'EXACTLY two sentences: problem (with key file) then impact' },
        },
        required: ['title', 'locations', 'severity', 'summary'],
      },
    },
  },
  required: ['findings'],
}

const synth = await agent(`${CONTEXT}

Synthesize the verified findings that meet the '${THRESHOLD}' bar into at most ${MAX_FINDINGS}, ranked by adjusted
severity then blast radius, preferring distinct structural issues over facets of one. For each write EXACTLY two
sentences (problem with key file; then why it matters) and keep file:line locations.

VERIFIED FINDINGS AT/ABOVE BAR (JSON):
${JSON.stringify(atBar.slice(0, Math.max(MAX_FINDINGS * 2, MAX_FINDINGS)).map(f => ({
    title: f.title, dimension: f.dimension, severity: f.sev, locations: f.locations,
    concern: f.concern, impact: f.impact, recommendation: f.recommendation,
    verification: f.verdict && f.verdict.assessment,
  })), null, 2)}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

return {
  threshold: THRESHOLD,
  converged: false,
  maxSeverity,
  findings: (synth.findings || []).slice(0, MAX_FINDINGS),
}
