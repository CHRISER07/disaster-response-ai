'use client'

import { useEffect, useState, useCallback } from 'react'
import type { Alert } from '@/types'
import { getAlerts, clearAlerts } from '@/lib/api'
import { SEVERITY_COLORS } from '@/lib/constants'

function AlertCard({ alert }: { alert: Alert }) {
  const colors = SEVERITY_COLORS[alert.severity] ?? SEVERITY_COLORS.LOW
  const ts = alert.timestamp ? alert.timestamp.slice(0, 16).replace('T', ' ') : ''

  return (
    <div className={`${colors.bg} ${colors.border} border-l-2 rounded-r-xl px-3 py-3 fade-in`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`text-[9px] font-bold ${colors.text} tracking-widest`}>
          {alert.severity}
        </span>
        <span className="text-[9px] text-slate-600 mono">{ts}</span>
      </div>
      {alert.zone && (
        <div className="text-[10px] text-slate-500 mono mb-1">{alert.zone}</div>
      )}
      <p className="text-[11px] text-slate-300 leading-snug line-clamp-3">{alert.message}</p>
      {alert.confidence && (
        <div className="mt-1.5">
          <span className="text-[9px] px-1.5 py-0.5 bg-[#1e2d3d] text-slate-500 rounded mono">
            {alert.confidence}
          </span>
        </div>
      )}
    </div>
  )
}

interface AlertPanelProps {
  alerts?: Alert[]
  loading?: boolean
  onClear?: () => void
}

export function AlertPanel({ alerts: propAlerts, loading: propLoading, onClear: propOnClear }: AlertPanelProps = {}) {
  const [internalAlerts, setInternalAlerts] = useState<Alert[]>([])
  const [internalLoading, setInternalLoading] = useState(true)

  const alerts = propAlerts !== undefined ? propAlerts : internalAlerts
  const loading = propLoading !== undefined ? propLoading : internalLoading

  const fetchAlerts = useCallback(async () => {
    try {
      const data = await getAlerts()
      setInternalAlerts(data.alerts)
    } catch {
      // API might be offline, silent fail
    } finally {
      setInternalLoading(false)
    }
  }, [])

  useEffect(() => {
    if (propAlerts === undefined) {
      fetchAlerts()
      const interval = setInterval(fetchAlerts, 15000)
      return () => clearInterval(interval)
    }
  }, [fetchAlerts, propAlerts])

  const handleClear = async () => {
    if (propOnClear) {
      propOnClear()
    } else {
      await clearAlerts()
      setInternalAlerts([])
    }
  }

  const criticalCount = alerts.filter(a => a.severity === 'CRITICAL').length

  return (
    <div className="border-l border-[#1e2d3d] bg-[#080c14]/60 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e2d3d] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-300 tracking-wide uppercase">Alerts</span>
          {criticalCount > 0 && (
            <span className="w-4 h-4 bg-red-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center animate-pulse">
              {criticalCount}
            </span>
          )}
          {alerts.length > 0 && (
            <span className="text-[9px] text-slate-600">{alerts.length} total</span>
          )}
        </div>
        {alerts.length > 0 && (
          <button
            onClick={handleClear}
            className="text-[10px] text-slate-600 hover:text-red-400 transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-600">
            Loading alerts...
          </div>
        ) : alerts.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <div className="w-10 h-10 rounded-full bg-emerald-600/10 border border-emerald-500/20 flex items-center justify-center mb-3">
              <svg className="w-5 h-5 text-emerald-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="text-xs font-medium text-emerald-400">All Clear</div>
            <div className="text-[10px] text-slate-600 mt-1">
              No alerts active. Alerts appear when the agent detects CRITICAL conditions.
            </div>
          </div>
        ) : (
          alerts.map((alert, i) => <AlertCard key={i} alert={alert} />)
        )}
      </div>

      {/* Auto-refresh indicator */}
      <div className="px-4 py-2 border-t border-[#1e2d3d] flex-shrink-0">
        <div className="flex items-center gap-1.5 text-[9px] text-slate-600">
          <span className="w-1.5 h-1.5 bg-slate-700 rounded-full animate-pulse" />
          Auto-refresh every 15s
        </div>
      </div>
    </div>
  )
}
