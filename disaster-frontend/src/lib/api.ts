/**
 * lib/api.ts
 * Typed API client for the Disaster Response FastAPI backend.
 * All calls route through Next.js /api/backend/* rewrite to avoid CORS.
 */

import type {
  AgentQueryRequest,
  AgentQueryResponse,
  Alert,
  EvalResult,
  EvalSummary,
  HealthStatus,
} from '@/types'

const BASE = '/api/backend'

async function fetchJSON<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...opts?.headers },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

/* Health */
export async function getHealth(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>('/health/')
}

export async function getReadiness(): Promise<HealthStatus> {
  return fetchJSON<HealthStatus>('/health/ready')
}

export async function getEnvStatus(): Promise<Record<string, string>> {
  return fetchJSON<Record<string, string>>('/health/env')
}

/* Agent */
export async function queryAgent(req: AgentQueryRequest): Promise<AgentQueryResponse> {
  return fetchJSON<AgentQueryResponse>('/agent/query', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

/**
 * Streams tokens and tool execution notifications from the agent via SSE.
 */
export async function streamAgent(
  req: AgentQueryRequest,
  onChunk: (chunk: string) => void,
  onToolCall: (tool: string) => void,
  onError: (err: string) => void,
  onDone: (finalAnswer?: string) => void
): Promise<void> {
  const response = await fetch(`${BASE}/agent/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

  if (!response.ok || !response.body) {
    throw new Error(`Stream connection failed with status ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      const lines = buffer.split('\n\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6))
            if (data.type === 'chunk' && data.content) {
              // Live-stream partial content as it arrives
              onChunk(data.content)
            } else if (data.type === 'tool_call' && data.tool) {
              onToolCall(data.tool)
            } else if (data.type === 'error') {
              onError(data.detail || 'Streaming error occurred.')
            } else if (data.type === 'done') {
              // `final_answer` is the DSEL-validated, safety-repaired final answer.
              // Always prefer it over any intermediate chunk content.
              onDone(data.final_answer || undefined)
            }
          } catch {
            // Ignore malformed intermediate JSON frames
          }
        }
      }
    }
  } catch (err) {
    onError(String(err))
  } finally {
    onDone()
  }
}

export async function getHistory(threadId: string): Promise<{ messages: unknown[]; count: number }> {
  return fetchJSON(`/agent/history/${threadId}`)
}

export async function clearHistory(threadId: string): Promise<{ status: string }> {
  return fetchJSON(`/agent/history/${threadId}`, { method: 'DELETE' })
}

/* Alerts */
export async function getAlerts(): Promise<{ alerts: Alert[]; count: number }> {
  return fetchJSON<{ alerts: Alert[]; count: number }>('/eval/alerts')
}

export async function clearAlerts(): Promise<{ status: string }> {
  return fetchJSON('/eval/alerts', { method: 'DELETE' })
}

/* Evaluation */
export async function getEvalResults(): Promise<{ results: EvalResult[]; count: number; message?: string }> {
  return fetchJSON('/eval/results')
}

export async function getEvalSummary(): Promise<{ summary: EvalSummary }> {
  return fetchJSON('/eval/summary')
}

/* Live sensor data (direct to Open-Meteo / USGS — no API key needed) */
export async function getLiveWeather(lat: number, lon: number) {
  const url = new URL('https://api.open-meteo.com/v1/forecast')
  url.searchParams.set('latitude', String(lat))
  url.searchParams.set('longitude', String(lon))
  url.searchParams.set('current', 'temperature_2m,rain,wind_speed_10m,weather_code,relative_humidity_2m,apparent_temperature')
  url.searchParams.set('timezone', 'auto')

  const res = await fetch(url.toString(), { next: { revalidate: 60 } })
  if (!res.ok) throw new Error('Weather fetch failed')
  const data = await res.json()
  return data.current as {
    temperature_2m: number
    rain: number
    wind_speed_10m: number
    weather_code: number
    relative_humidity_2m: number
    apparent_temperature: number
  }
}

export async function getLiveGauge(siteId: string) {
  const url = new URL('https://waterservices.usgs.gov/nwis/iv/')
  url.searchParams.set('sites', siteId)
  url.searchParams.set('parameterCd', '00065')
  url.searchParams.set('format', 'json')

  const res = await fetch(url.toString(), { next: { revalidate: 60 } })
  if (!res.ok) throw new Error('USGS fetch failed')
  const json = await res.json()
  const ts = json?.value?.timeSeries?.[0]
  if (!ts) throw new Error('No timeSeries in USGS response')

  const values = ts?.values?.[0]?.value ?? []
  if (!values.length) throw new Error('No gauge values')

  const latest = values[values.length - 1]
  return {
    gage: parseFloat(latest.value),
    time: latest.dateTime as string,
    site: ts.sourceInfo.siteName as string,
  }
}

export async function geocodeCity(city: string): Promise<{ lat: number; lon: number; name: string } | null> {
  const url = new URL('https://geocoding-api.open-meteo.com/v1/search')
  url.searchParams.set('name', city)
  url.searchParams.set('count', '1')
  url.searchParams.set('language', 'en')
  url.searchParams.set('format', 'json')

  const res = await fetch(url.toString())
  if (!res.ok) return null
  const data = await res.json()
  const r = data?.results?.[0]
  if (!r) return null
  return {
    lat: r.latitude as number,
    lon: r.longitude as number,
    name: `${r.name}, ${r.country_code}` as string,
  }
}
