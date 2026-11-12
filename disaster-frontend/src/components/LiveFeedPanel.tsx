'use client'

import { useEffect, useState, useCallback } from 'react'
import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from 'recharts'
import type { LocationConfig } from '@/types'
import { getLiveWeather, getLiveGauge, geocodeCity } from '@/lib/api'
import { WMO_CODES, LOCATION_PRESETS } from '@/lib/constants'

interface LiveFeedPanelProps {
  location: LocationConfig
  onLocationChange: (loc: LocationConfig) => void
}

function GaugeChart({ value, max, danger, label, unit, color }: {
  value: number; max: number; danger: number; label: string; unit: string; color: string
}) {
  const pct = Math.min((value / max) * 100, 100)
  const data = [{ value: pct, fill: color }]

  return (
    <div className="relative flex flex-col items-center">
      <div className="w-full h-28">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%" cy="85%"
            innerRadius="60%" outerRadius="90%"
            startAngle={180} endAngle={0}
            data={data}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" cornerRadius={4} background={{ fill: '#1e2d3d' }} />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="text-center -mt-6">
        <div className="text-lg font-bold mono" style={{ color }}>
          {value.toFixed(2)}
        </div>
        <div className="text-[10px] text-slate-500">{unit}</div>
        <div className="text-[10px] text-slate-400 mt-0.5">{label}</div>
      </div>
    </div>
  )
}

function getSeverityInfo(gage: number, threshold: number): { label: string; color: string; bg: string } {
  if (gage >= threshold * 1.3) return { label: 'CRITICAL',    color: '#ef4444', bg: 'bg-red-950/40' }
  if (gage >= threshold)       return { label: 'DANGER',      color: '#f97316', bg: 'bg-orange-950/40' }
  if (gage >= threshold * 0.8) return { label: 'WARNING',     color: '#f59e0b', bg: 'bg-amber-950/40' }
  return                              { label: 'NORMAL',      color: '#10b981', bg: 'bg-emerald-950/40' }
}

function WeatherSeverity(rain: number, wind: number): string {
  if (rain > 10 || wind > 80) return 'CRITICAL'
  if (rain > 5  || wind > 60) return 'HIGH'
  if (rain > 1  || wind > 40) return 'MODERATE'
  return 'NORMAL'
}

export function LiveFeedPanel({ location, onLocationChange }: LiveFeedPanelProps) {
  const [mounted, setMounted] = useState(false)
  const [weather, setWeather] = useState<{
    temp: number; rain: number; wind: number
    humidity: number; feels_like: number; code: number
  } | null>(null)

  const [gauge, setGauge] = useState<{ gage: number; time: string; site: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())

  // Location selector states
  const [selectedPreset, setSelectedPreset] = useState('Boulder, CO (2013 Flood)')
  const [customCity, setCustomCity] = useState('')
  const [geocoding, setGeocoding] = useState(false)
  const [geoError, setGeoError] = useState('')

  const fetchData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [w, g] = await Promise.allSettled([
        getLiveWeather(location.lat, location.lon),
        getLiveGauge(location.usgs),
      ])

      if (w.status === 'fulfilled') {
        setWeather({
          temp: w.value.temperature_2m,
          rain: w.value.rain,
          wind: w.value.wind_speed_10m,
          humidity: w.value.relative_humidity_2m,
          feels_like: w.value.apparent_temperature,
          code: w.value.weather_code,
        })
      }
      if (g.status === 'fulfilled') setGauge(g.value)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
      setLastRefresh(new Date())
    }
  }, [location])

  useEffect(() => {
    setMounted(true)
    fetchData()
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [fetchData])

  const handlePresetChange = (presetName: string) => {
    setSelectedPreset(presetName)
    setGeoError('')
    if (presetName !== 'Custom Location') {
      onLocationChange(LOCATION_PRESETS[presetName])
    }
  }

  const handleCustomSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!customCity.trim()) return
    setGeocoding(true)
    setGeoError('')
    try {
      const geo = await geocodeCity(customCity.trim())
      if (geo) {
        onLocationChange({
          city: geo.name,
          usgs: '06730200', // Default USGS, tools geocode to fallback automatically
          lat: geo.lat,
          lon: geo.lon,
          threshold: 10.0,
        })
      } else {
        setGeoError(`Could not find "${customCity}"`)
      }
    } catch {
      setGeoError('Geocoding failed')
    } finally {
      setGeocoding(false)
    }
  }

  const sev = gauge ? getSeverityInfo(gauge.gage, location.threshold) : null
  const wSev = weather ? WeatherSeverity(weather.rain, weather.wind) : 'NORMAL'

  return (
    <div className="border-r border-[#1e2d3d] bg-[#080c14]/60 flex flex-col overflow-y-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e2d3d] flex-shrink-0">
        <span className="text-xs font-semibold text-slate-300 tracking-wide uppercase">Live Feed</span>
        <button
          onClick={fetchData}
          className="text-[10px] text-slate-500 hover:text-blue-400 transition-colors mono"
        >
          {loading ? 'Loading...' : `Refresh | ${mounted ? lastRefresh.toLocaleTimeString() : ''}`}
        </button>
      </div>

      <div className="flex-1 p-4 space-y-4">
        {/* Location Picker Section */}
        <div className="glass rounded-xl p-3 space-y-2">
          <label className="text-[9px] font-semibold text-slate-500 uppercase tracking-widest block">
            Target Location
          </label>
          <select
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="w-full bg-[#0d1117] border border-[#1e2d3d] rounded px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500/60"
          >
            {Object.keys(LOCATION_PRESETS).map(key => (
              <option key={key} value={key}>{key}</option>
            ))}
            <option value="Custom Location">Custom Location...</option>
          </select>

          {selectedPreset === 'Custom Location' && (
            <form onSubmit={handleCustomSubmit} className="pt-1">
              <input
                type="text"
                value={customCity}
                onChange={e => setCustomCity(e.target.value)}
                placeholder="e.g. Caracas, Venezuela"
                className="w-full bg-[#0d1117] border border-[#1e2d3d] rounded px-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-500/60"
              />
              <button
                type="submit"
                disabled={geocoding}
                className="mt-1.5 w-full py-1 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/30 rounded text-[11px] font-medium transition-all disabled:opacity-50"
              >
                {geocoding ? 'Locating...' : 'Set Monitor Location'}
              </button>
              {geoError && <p className="mt-1 text-[10px] text-red-400">{geoError}</p>}
            </form>
          )}

          {/* Mini info card */}
          <div className="bg-[#0d1117]/40 rounded p-2 mt-1 space-y-0.5 text-[9px] text-slate-500 mono border border-[#1e2d3d]/50">
            <div>Coords: {location.lat.toFixed(4)}, {location.lon.toFixed(4)}</div>
            <div>USGS Gauge: #{location.usgs}</div>
          </div>
        </div>

        {/* River gauge */}
        <div className="glass rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">River Gage Height</span>
            {sev && (
              <span
                className="text-[9px] font-bold px-2 py-0.5 rounded-full border"
                style={{ color: sev.color, borderColor: sev.color + '50', backgroundColor: sev.color + '15' }}
              >
                {sev.label}
              </span>
            )}
          </div>

          {gauge ? (
            <>
              <GaugeChart
                value={gauge.gage}
                max={location.threshold * 2}
                danger={location.threshold}
                label="Gage Height"
                unit="ft"
                color={sev?.color ?? '#3b82f6'}
              />
              <div className="mt-3 space-y-1 text-[10px]">
                <div className="flex justify-between">
                  <span className="text-slate-500">Site</span>
                  <span className="text-slate-300 mono truncate max-w-[140px]">{gauge.site}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Flood Stage</span>
                  <span className="text-slate-300 mono">{location.threshold} ft</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Updated</span>
                  <span className="text-slate-300 mono">{gauge.time.slice(0, 16).replace('T', ' ')}</span>
                </div>
              </div>
            </>
          ) : (
            <div className="h-28 flex items-center justify-center text-xs text-slate-600">
              {loading ? 'Fetching USGS data...' : 'USGS unavailable for this site'}
            </div>
          )}
        </div>

        {/* Weather card */}
        <div className="glass rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">Weather</span>
            <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
              wSev === 'CRITICAL' ? 'text-red-400 border-red-500/40 bg-red-500/10' :
              wSev === 'HIGH'     ? 'text-orange-400 border-orange-500/40 bg-orange-500/10' :
              wSev === 'MODERATE' ? 'text-amber-400 border-amber-500/40 bg-amber-500/10' :
              'text-emerald-400 border-emerald-500/40 bg-emerald-500/10'
            }`}>{wSev}</span>
          </div>

          {weather ? (
            <div className="space-y-2">
              <div className="flex items-end gap-1">
                <span className="text-3xl font-bold text-white">{weather.temp.toFixed(1)}</span>
                <span className="text-slate-400 mb-1 text-sm">deg C</span>
              </div>
              <div className="text-[10px] text-slate-500">
                Feels like {weather.feels_like.toFixed(1)} deg C
              </div>
              <div className="text-xs text-slate-400">{WMO_CODES[weather.code] ?? 'Unknown'}</div>

              <div className="grid grid-cols-3 gap-2 mt-3">
                {[
                  { label: 'Rain', value: `${weather.rain.toFixed(1)}mm/h` },
                  { label: 'Wind', value: `${weather.wind.toFixed(0)}km/h` },
                  { label: 'Humidity', value: `${weather.humidity}%` },
                ].map(m => (
                  <div key={m.label} className="bg-[#0d1117]/80 rounded p-2 text-center">
                    <div className="mono text-[11px] font-semibold text-slate-200">{m.value}</div>
                    <div className="text-[9px] text-slate-600 mt-0.5">{m.label}</div>
                  </div>
                ))}
              </div>

              <div className="text-[9px] text-slate-600 mt-2">
                {location.city} | Open-Meteo
              </div>
            </div>
          ) : (
            <div className="h-28 flex items-center justify-center text-xs text-slate-600">
              {loading ? 'Fetching weather...' : 'Weather unavailable'}
            </div>
          )}
        </div>

        {/* Quick Links Section at the bottom */}
        <div className="glass rounded-xl p-3">
          <span className="text-[9px] font-semibold text-slate-500 uppercase tracking-widest block mb-2">
            API Links
          </span>
          <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[#0d1117]/60 hover:bg-blue-600/10 text-slate-400 hover:text-blue-400 border border-[#1e2d3d] rounded py-1 transition-all"
            >
              Swagger
            </a>
            <a
              href="http://localhost:8000/health/"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[#0d1117]/60 hover:bg-blue-600/10 text-slate-400 hover:text-blue-400 border border-[#1e2d3d] rounded py-1 transition-all"
            >
              Health
            </a>
            <a
              href="https://smith.langchain.com"
              target="_blank"
              rel="noopener noreferrer"
              className="bg-[#0d1117]/60 hover:bg-blue-600/10 text-slate-400 hover:text-blue-400 border border-[#1e2d3d] rounded py-1 transition-all"
            >
              LangSmith
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
