// Shared TypeScript types for the Disaster Response AI frontend

export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NORMAL' | 'WARNING' | 'DANGER'

export interface Alert {
  severity: Severity
  message: string
  zone: string
  timestamp: string
  tool?: string
  confidence?: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  tool_calls?: string[]
  cost_usd?: number
  latency_s?: number
  timestamp?: string
}

export interface AgentQueryRequest {
  query: string
  thread_id: string
  location_context?: string
}

export interface AgentQueryResponse {
  answer: string
  tool_calls: string[]
  thread_id: string
  latency_s: number
  cost_usd?: number
}

export interface WeatherData {
  temp: number
  rain: number
  wind: number
  humidity?: number
  feels_like?: number
  condition?: string
  severity?: Severity
}

export interface SensorData {
  gage: number
  threshold: number
  status: Severity
  site: string
  time: string
  pct_above: number
}

export interface EarthquakeEvent {
  magnitude: number
  place: string
  time: string
}

export interface LocationConfig {
  city: string
  usgs: string
  lat: number
  lon: number
  threshold: number
}

export interface HealthStatus {
  status: 'ok' | 'not_ready'
  doc_count?: number
  reason?: string
}

export interface CostRecord {
  ts: string
  query_id: string
  thread_id: string
  model: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cost_usd: number
  latency_s: number
  tool_calls: string[]
}

export interface EvalResult {
  query_id: string
  config: string
  query: string
  kw_f1: string
  dr_fact: string
  saa: string
  p_at_k: string
  latency_s: string
  svc: string
  eval_type: string
}

export interface EvalSummary {
  [config: string]: {
    n: number
    kw_f1: number
    dr_fact: number
    saa: number
    p_at_k: number
    latency_s: number
    svc_total: number
  }
}

export interface PolicyViolation {
  ts: string
  tool: string
  allowed: boolean
  reason: string
  confidence: string
  args_summary: Record<string, string>
}
