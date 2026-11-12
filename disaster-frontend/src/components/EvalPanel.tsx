'use client'

import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts'
import type { EvalSummary, EvalResult } from '@/types'
import { getEvalSummary, getEvalResults } from '@/lib/api'

const CONFIG_COLORS: Record<string, string> = {
  'multimodal_rag': '#3b82f6',
  'text_rag':       '#06b6d4',
  'llm_only':       '#8b5cf6',
  'bm25_baseline':  '#6b7280',
}

const METRIC_LABELS: Record<string, string> = {
  kw_f1:     'KW-F1',
  dr_fact:   'DR-Fact',
  saa:       'SAA',
  p_at_k:    'P@K',
  latency_s: 'Latency (s)',
}

const DISASTER_TYPE_COLORS: Record<string, string> = {
  flood:             '#3b82f6',
  earthquake:        '#f59e0b',
  wildfire:          '#ef4444',
  typhoon:           '#8b5cf6',
  building_collapse: '#f97316',
  chemical_hazard:   '#10b981',
  general:           '#6b7280',
}

export function EvalPanel() {
  const [summary, setSummary] = useState<EvalSummary | null>(null)
  const [results, setResults] = useState<EvalResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeMetric, setActiveMetric] = useState<string>('kw_f1')
  const [filterConfig, setFilterConfig] = useState<string>('all')
  const [filterType, setFilterType] = useState<string>('all')

  useEffect(() => {
    const load = async () => {
      try {
        const [s, r] = await Promise.all([getEvalSummary(), getEvalResults()])
        setSummary(s.summary)
        setResults(r.results)
      } catch (e) {
        setError(String(e))
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
        Loading evaluation results...
      </div>
    )
  }

  if (error || !summary) {
    return (
      <div className="glass rounded-xl p-8 text-center">
        <div className="text-red-400 text-sm mb-2">Evaluation results not found</div>
        <div className="text-slate-500 text-xs">
          Run <span className="mono bg-[#0d1117] px-2 py-0.5 rounded">venv\Scripts\python.exe evaluation\run_evals.py</span> first.
        </div>
      </div>
    )
  }

  const configs = Object.keys(summary)

  // Build chart data
  const chartData = configs.map(cfg => ({
    config: cfg.replace('_', ' ').toUpperCase(),
    configKey: cfg,
    value: summary[cfg][activeMetric as keyof typeof summary[string]] as number,
    fill: CONFIG_COLORS[cfg] ?? '#6b7280',
  }))

  // Filter results
  const filteredResults = results.filter(r => {
    if (filterConfig !== 'all' && r.config !== filterConfig) return false
    return true
  })

  return (
    <div className="max-w-6xl mx-auto space-y-6 fade-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">Evaluation Results</h1>
        <p className="text-sm text-slate-500 mt-1">
          Dual-metric framework: static KW-F1 + LLM-as-Judge vs live behavioral citation checks
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {configs.map(cfg => {
          const s = summary[cfg]
          const color = CONFIG_COLORS[cfg] ?? '#6b7280'
          return (
            <div key={cfg} className="glass rounded-xl p-4 border-t-2" style={{ borderTopColor: color }}>
              <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest mb-3">
                {cfg.replace(/_/g, ' ')}
              </div>
              <div className="space-y-1">
                {['kw_f1', 'dr_fact', 'saa'].map(m => (
                  <div key={m} className="flex justify-between text-xs">
                    <span className="text-slate-500">{METRIC_LABELS[m]}</span>
                    <span className="mono font-semibold text-slate-200">
                      {(s[m as keyof typeof s] as number).toFixed(3)}
                    </span>
                  </div>
                ))}
                <div className="flex justify-between text-xs pt-1 border-t border-[#1e2d3d]">
                  <span className="text-slate-500">SVC</span>
                  <span className={`mono font-bold ${s.svc_total > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                    {s.svc_total}
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Avg Latency</span>
                  <span className="mono text-slate-400">{s.latency_s.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Queries</span>
                  <span className="mono text-slate-400">{s.n}</span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Bar chart */}
      <div className="glass rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm font-semibold text-slate-300">Ablation Comparison</div>
          <div className="flex gap-2">
            {Object.entries(METRIC_LABELS).slice(0, 4).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveMetric(key)}
                className={`px-3 py-1 text-[11px] rounded transition-all ${
                  activeMetric === key
                    ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30'
                    : 'text-slate-500 hover:text-slate-300 border border-transparent'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
              <XAxis dataKey="config" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ background: '#0f1923', border: '1px solid #1e2d3d', borderRadius: '8px', color: '#e2e8f0', fontSize: '12px' }}
              />
              <Bar dataKey="value" name={METRIC_LABELS[activeMetric]}>
                {chartData.map((entry, idx) => (
                  <Cell key={`cell-${idx}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Per-query table */}
      {results.length > 0 && (
        <div className="glass rounded-xl overflow-hidden">
          <div className="flex items-center justify-between p-4 border-b border-[#1e2d3d]">
            <div className="text-sm font-semibold text-slate-300">Per-Query Results</div>
            <div className="flex gap-2">
              <select
                value={filterConfig}
                onChange={e => setFilterConfig(e.target.value)}
                className="bg-[#0d1117] border border-[#1e2d3d] rounded px-2 py-1 text-[11px] text-slate-400"
              >
                <option value="all">All configs</option>
                {configs.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#1e2d3d]">
                  {['Config', 'Query', 'Type', 'KW-F1', 'DR-Fact', 'SAA', 'SVC', 'Latency'].map(h => (
                    <th key={h} className="text-left px-4 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredResults.slice(0, 50).map((r, i) => (
                  <tr key={i} className="border-b border-[#1e2d3d]/50 hover:bg-white/2 transition-colors">
                    <td className="px-4 py-2">
                      <span className="mono text-[10px]" style={{ color: CONFIG_COLORS[r.config] ?? '#6b7280' }}>
                        {r.config}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-slate-300 max-w-[200px] truncate">{r.query}</td>
                    <td className="px-4 py-2">
                      <span className={`text-[9px] px-1.5 py-0.5 rounded ${
                        r.eval_type === 'behavioral' ? 'bg-purple-500/20 text-purple-300' : 'bg-blue-500/20 text-blue-300'
                      }`}>
                        {r.eval_type}
                      </span>
                    </td>
                    <td className="px-4 py-2 mono text-slate-300">{parseFloat(r.kw_f1 || '0').toFixed(2)}</td>
                    <td className="px-4 py-2 mono text-slate-300">{parseFloat(r.dr_fact || '0').toFixed(2)}</td>
                    <td className="px-4 py-2 mono text-slate-300">{parseFloat(r.saa || '0').toFixed(2)}</td>
                    <td className="px-4 py-2 mono">
                      <span className={parseInt(r.svc || '0') > 0 ? 'text-red-400' : 'text-emerald-400'}>
                        {r.svc}
                      </span>
                    </td>
                    <td className="px-4 py-2 mono text-slate-400">{parseFloat(r.latency_s || '0').toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filteredResults.length > 50 && (
              <div className="px-4 py-2 text-[10px] text-slate-600">
                Showing first 50 of {filteredResults.length} results
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
