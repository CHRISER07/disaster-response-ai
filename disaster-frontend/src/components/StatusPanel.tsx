'use client'

import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { PolicyViolation, CostRecord } from '@/types'

interface StatusPanelProps {
  agentOnline: boolean | null
  dbDocCount: number
}

function StatusBadge({ ok }: { ok: boolean | null }) {
  if (ok === null) return <span className="text-[10px] text-amber-400 mono">CHECKING</span>
  return ok
    ? <span className="text-[10px] text-emerald-400 mono">ONLINE</span>
    : <span className="text-[10px] text-red-400 mono">OFFLINE</span>
}

function StatCard({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="glass rounded-xl p-4">
      <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">{label}</div>
      <div className="text-2xl font-bold" style={{ color: color ?? '#e2e8f0' }}>{value}</div>
      {sub && <div className="text-[10px] text-slate-600 mt-1 mono">{sub}</div>}
    </div>
  )
}

export function StatusPanel({ agentOnline, dbDocCount }: StatusPanelProps) {
  const [envStatus, setEnvStatus] = useState<Record<string, string>>({})
  const [readiness, setReadiness] = useState<{ status: string; doc_count?: number; reason?: string } | null>(null)
  const [costRecords, setCostRecords] = useState<CostRecord[]>([])
  const [violations, setViolations] = useState<PolicyViolation[]>([])
  const [loadingStatus, setLoadingStatus] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [env, ready] = await Promise.all([
          fetch('/api/backend/health/env').then(r => r.json()),
          fetch('/api/backend/health/ready').then(r => r.json()),
        ])
        setEnvStatus(env)
        setReadiness(ready)
      } catch {
        // API offline
      } finally {
        setLoadingStatus(false)
      }
    }
    load()
  }, [])

  // Session metrics summary from cost endpoint if available
  const totalCost = costRecords.reduce((acc, r) => acc + (r.cost_usd ?? 0), 0)
  const totalTokens = costRecords.reduce((acc, r) => acc + (r.total_tokens ?? 0), 0)
  const avgLatency = costRecords.length > 0
    ? (costRecords.reduce((acc, r) => acc + r.latency_s, 0) / costRecords.length)
    : 0

  const costChartData = costRecords.slice(-20).map((r, i) => ({
    i: i + 1,
    cost: +(r.cost_usd * 1000).toFixed(3),
    latency: +r.latency_s.toFixed(1),
  }))

  return (
    <div className="max-w-5xl mx-auto space-y-6 fade-in">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">System Status</h1>
        <p className="text-sm text-slate-500 mt-1">Service health, cost tracking, and policy audit trail</p>
      </div>

      {/* Service health cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass rounded-xl p-4">
          <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">FastAPI Backend</div>
          <div className="flex items-center gap-2">
            <span className={`pulse-dot ${agentOnline === null ? 'amber' : agentOnline ? 'green' : 'red'}`} />
            <StatusBadge ok={agentOnline} />
          </div>
          <div className="text-[10px] text-slate-600 mt-1 mono">:8000</div>
        </div>

        <div className="glass rounded-xl p-4">
          <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2">Vector DB</div>
          <div className="text-lg font-bold text-white">{dbDocCount.toLocaleString()}</div>
          <div className="text-[10px] text-slate-600 mt-1 mono">
            {readiness?.status === 'ready' ? 'ChromaDB Ready' : readiness?.reason ?? 'Checking...'}
          </div>
        </div>

        {Object.entries(envStatus).map(([key, val]) => (
          <div key={key} className="glass rounded-xl p-4">
            <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-2 truncate">
              {key.replace('LANGCHAIN_', '').replace('_', ' ')}
            </div>
            <div className={`text-sm font-semibold ${val === 'set' ? 'text-emerald-400' : 'text-red-400'}`}>
              {val === 'set' ? 'Configured' : 'Missing'}
            </div>
          </div>
        ))}
      </div>

      {/* Architecture overview */}
      <div className="glass rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-4">System Architecture</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-[11px]">
          {[
            {
              title: 'Frontend (This App)',
              items: ['Next.js 15 + TypeScript', 'Tailwind CSS + Recharts', 'Framer Motion animations', 'Port 3000'],
              color: '#3b82f6',
            },
            {
              title: 'FastAPI Backend',
              items: ['LangGraph ReAct Agent', 'Llama 3.3 70B via Groq', '8 tools + ChromaDB', 'Port 8000'],
              color: '#06b6d4',
            },
            {
              title: 'Data & Observability',
              items: ['16,849 CrisisLex tweets', 'USGS + Open-Meteo APIs', 'LangSmith tracing', 'Cost log JSONL'],
              color: '#10b981',
            },
          ].map(layer => (
            <div key={layer.title} className="bg-[#0d1117]/60 rounded-xl p-4 border-t-2" style={{ borderTopColor: layer.color }}>
              <div className="font-semibold text-slate-200 mb-2" style={{ color: layer.color }}>{layer.title}</div>
              <ul className="space-y-1">
                {layer.items.map(item => (
                  <li key={item} className="flex items-center gap-2 text-slate-400">
                    <span className="w-1 h-1 rounded-full flex-shrink-0" style={{ backgroundColor: layer.color }} />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Cost chart placeholder */}
      <div className="glass rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-2">Session Cost History</div>
        <p className="text-[11px] text-slate-500 mb-4">
          Cost data is stored in <span className="mono bg-[#0d1117] px-1.5 py-0.5 rounded">logs/cost_log.jsonl</span> by the FastAPI backend.
          Run queries via the Command Center to populate cost data.
        </p>

        {costChartData.length > 0 ? (
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={costChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2d3d" />
                <XAxis dataKey="i" tick={{ fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#0f1923', border: '1px solid #1e2d3d', borderRadius: '8px', color: '#e2e8f0', fontSize: '12px' }}
                  formatter={(v) => [`$${(Number(v ?? 0) / 1000).toFixed(4)}`, 'Cost']}
                />
                <Line type="monotone" dataKey="cost" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-24 flex items-center justify-center text-xs text-slate-600">
            No cost data yet - send queries to populate this chart
          </div>
        )}
      </div>

      {/* Key model info */}
      <div className="glass rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-4">Model Configuration</div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
          {[
            { label: 'LLM',        value: 'llama-3.3-70b-versatile' },
            { label: 'Provider',   value: 'Groq (free tier)' },
            { label: 'Embeddings', value: 'BAAI/bge-small-en-v1.5' },
            { label: 'Vision',     value: 'Ollama LLaVA 7B (local)' },
            { label: 'Vector DB',  value: 'ChromaDB persistent' },
            { label: 'Agentic',    value: 'LangGraph create_react_agent' },
          ].map(item => (
            <div key={item.label} className="bg-[#0d1117]/60 rounded-lg px-3 py-2">
              <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">{item.label}</div>
              <div className="mono text-slate-200 text-[11px]">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Safety validator info */}
      <div className="glass rounded-xl p-6">
        <div className="text-sm font-semibold text-slate-300 mb-2">Safety & Policy</div>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Safety Validator</div>
            <div className="space-y-1 text-slate-400">
              <div>14 regex patterns enforced post-generation</div>
              <div>Categories: flood, HAZMAT, chemical, re-entry</div>
              <div>Zero LLM inference required - deterministic</div>
            </div>
          </div>
          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Policy Engine</div>
            <div className="space-y-1 text-slate-400">
              <div>Read tools: always allowed (sensors, KB, weather)</div>
              <div>Write tools: gated by [Confidence: HIGH/MEDIUM]</div>
              <div>Violations logged to logs/policy_violations.jsonl</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
