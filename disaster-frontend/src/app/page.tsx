'use client'

import { useState, useEffect, useCallback } from 'react'
import { TopBar } from '@/components/TopBar'
import { LiveFeedPanel } from '@/components/LiveFeedPanel'
import { ChatPanel } from '@/components/ChatPanel'
import { AlertPanel } from '@/components/AlertPanel'
import { EvalPanel } from '@/components/EvalPanel'
import { StatusPanel } from '@/components/StatusPanel'
import type { LocationConfig, ChatMessage, Alert } from '@/types'
import { LOCATION_PRESETS, DEFAULT_THREAD_ID } from '@/lib/constants'
import { getAlerts, clearAlerts } from '@/lib/api'

export type ActiveTab = 'command' | 'evaluation' | 'status'

export default function Home() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('command')
  const [location, setLocation] = useState<LocationConfig>(LOCATION_PRESETS['Boulder, CO (2013 Flood)'])
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [agentOnline, setAgentOnline] = useState<boolean | null>(null)
  const [dbDocCount, setDbDocCount] = useState<number>(0)
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [alertsLoading, setAlertsLoading] = useState<boolean>(true)

  // Global Alerts polling
  const fetchGlobalAlerts = useCallback(async () => {
    try {
      const data = await getAlerts()
      setAlerts(data.alerts || [])
    } catch {
      // offline silent catch
    } finally {
      setAlertsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchGlobalAlerts()
    const alertInterval = setInterval(fetchGlobalAlerts, 10000)
    return () => clearInterval(alertInterval)
  }, [fetchGlobalAlerts])

  const handleClearAlerts = useCallback(async () => {
    await clearAlerts()
    setAlerts([])
  }, [])

  // Health check on mount
  useEffect(() => {
    const check = async () => {
      try {
        const [health, ready] = await Promise.all([
          fetch('/api/backend/health/').then(r => r.json()),
          fetch('/api/backend/health/ready').then(r => r.json()),
        ])
        setAgentOnline(health.status === 'ok')
        setDbDocCount(ready.doc_count ?? 0)
      } catch {
        setAgentOnline(false)
      }
    }
    check()
    const interval = setInterval(check, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleLocationChange = useCallback((loc: LocationConfig) => {
    setLocation(loc)
  }, [])

  const handleClearChat = useCallback(() => {
    setChatHistory([])
    fetch(`/api/backend/agent/history/${DEFAULT_THREAD_ID}`, { method: 'DELETE' }).catch(() => {})
  }, [])

  const criticalAlerts = alerts.filter(a => a.severity === 'CRITICAL')
  const criticalCount = criticalAlerts.length
  const latestCriticalAlert = criticalAlerts[0]

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <TopBar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        agentOnline={agentOnline}
        location={location.city}
        dbDocCount={dbDocCount}
        criticalCount={criticalCount}
      />

      {/* Global Critical Emergency Alert Banner */}
      {criticalCount > 0 && latestCriticalAlert && (
        <div className="bg-red-950/90 border-b border-red-500/60 px-4 py-2 flex items-center justify-between text-xs text-red-200 z-40 backdrop-blur-md">
          <div className="flex items-center gap-2 overflow-hidden">
            <span className="px-2 py-0.5 rounded bg-red-600 text-white font-bold text-[10px] tracking-widest uppercase animate-pulse flex-shrink-0">
              CRITICAL EMERGENCY
            </span>
            <span className="font-semibold text-white flex-shrink-0">{latestCriticalAlert.zone}:</span>
            <span className="truncate text-red-100">{latestCriticalAlert.message}</span>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0 ml-4">
            <span className="text-[10px] text-red-400 mono">
              {latestCriticalAlert.timestamp?.slice(11, 19)} UTC
            </span>
            {activeTab !== 'command' && (
              <button
                onClick={() => setActiveTab('command')}
                className="px-2 py-0.5 bg-red-700 hover:bg-red-600 text-white text-[10px] font-medium rounded transition-colors"
              >
                Go to Command Center &rarr;
              </button>
            )}
          </div>
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        <main className="flex-1 overflow-hidden">
          {activeTab === 'command' && (
            <div className="h-full grid grid-cols-[320px_1fr_320px] gap-0 overflow-hidden">
              <LiveFeedPanel location={location} onLocationChange={handleLocationChange} />
              <ChatPanel
                chatHistory={chatHistory}
                setChatHistory={setChatHistory}
                location={location}
                onClear={handleClearChat}
              />
              <AlertPanel
                alerts={alerts}
                loading={alertsLoading}
                onClear={handleClearAlerts}
              />
            </div>
          )}

          {activeTab === 'evaluation' && (
            <div className="h-full overflow-auto p-6">
              <EvalPanel />
            </div>
          )}

          {activeTab === 'status' && (
            <div className="h-full overflow-auto p-6">
              <StatusPanel agentOnline={agentOnline} dbDocCount={dbDocCount} />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
