'use client'

import { ActiveTab } from '@/app/page'
import { useState, useEffect } from 'react'

const TABS: { id: ActiveTab; label: string; icon: string }[] = [
  { id: 'command',    label: 'Command Center', icon: 'M' },
  { id: 'evaluation', label: 'Evaluation',     icon: 'E' },
  { id: 'status',     label: 'System Status',  icon: 'S' },
]

interface TopBarProps {
  activeTab: ActiveTab
  setActiveTab: (tab: ActiveTab) => void
  agentOnline: boolean | null
  location: string
  dbDocCount: number
  criticalCount?: number
}

export function TopBar({ activeTab, setActiveTab, agentOnline, location, dbDocCount, criticalCount = 0 }: TopBarProps) {
  const [timeString, setTimeString] = useState<string>('')
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  useEffect(() => {
    setTimeString(new Date().toUTCString().slice(0, 25))
    const timer = setInterval(() => {
      setTimeString(new Date().toUTCString().slice(0, 25))
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  // Sync theme with html root class
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'light') {
      root.classList.add('light')
    } else {
      root.classList.remove('light')
    }
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))
  }

  return (
    <header className="relative scanlines flex-shrink-0 border-b border-[#1e2d3d] bg-[#080c14]/95 backdrop-blur-md z-50">
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-blue-500 to-transparent opacity-60" />

      <div className="flex items-center justify-between px-4 py-3">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded bg-blue-600/20 border border-blue-500/30">
            <svg className="w-4 h-4 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <div className="text-sm font-bold text-white tracking-wide">DISASTER COMMAND CENTER</div>
            <div className="text-[10px] text-slate-500 mono">v2.0 AGENTIC RAG</div>
          </div>
        </div>

        {/* Nav Tabs */}
        <nav className="flex items-center gap-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-4 py-1.5 text-xs font-medium rounded transition-all duration-200 flex items-center gap-1.5 ${
                activeTab === tab.id
                  ? 'bg-blue-600/20 text-blue-300 border border-blue-500/40'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
              }`}
            >
              <span>{tab.label}</span>
              {tab.id === 'command' && criticalCount > 0 && (
                <span className="px-1.5 py-0.2 bg-red-600 text-white rounded-full text-[9px] font-bold animate-pulse">
                  {criticalCount}
                </span>
              )}
            </button>
          ))}
        </nav>

        {/* Status strip & Dark/Light Toggle */}
        <div className="flex items-center gap-3 text-[11px]">
          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} mode`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0f1923] hover:bg-[#1e2d3d] border border-[#1e2d3d] text-slate-300 text-[10px] transition-colors"
          >
            {theme === 'dark' ? (
              <>
                <svg className="w-3.5 h-3.5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                <span>Light</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
                <span>Dark</span>
              </>
            )}
          </button>

          <div className="flex items-center gap-1.5">
            <span
              className={`pulse-dot ${agentOnline === null ? 'amber' : agentOnline ? 'green' : 'red'}`}
            />
            <span className="text-slate-400">
              {agentOnline === null ? 'CHECKING' : agentOnline ? 'AGENT ONLINE' : 'AGENT OFFLINE'}
            </span>
          </div>
          <div className="w-px h-3 bg-slate-700" />
          <span className="text-slate-400 mono">{location}</span>
          <div className="w-px h-3 bg-slate-700" />
          <span className="text-slate-500 mono">{dbDocCount.toLocaleString()} docs</span>
          <div className="w-px h-3 bg-slate-700" />
          <span className="text-slate-600 mono">{timeString ? `${timeString} UTC` : 'LOADING...'}</span>
        </div>
      </div>
    </header>
  )
}
