/**
 * lib/constants.ts
 * Location presets, WMO weather codes, severity colors
 */

import type { LocationConfig, Severity } from '@/types'

export const LOCATION_PRESETS: Record<string, LocationConfig> = {
  'Boulder, CO (2013 Flood)': { city: 'Boulder, CO',    usgs: '06730200', lat: 40.0150,  lon: -105.2705, threshold: 10.0 },
  'Houston, TX':              { city: 'Houston, TX',     usgs: '08074000', lat: 29.7604,  lon: -95.3698,  threshold: 20.0 },
  'New Orleans, LA':          { city: 'New Orleans, LA', usgs: '07374000', lat: 29.9511,  lon: -90.0715,  threshold: 17.0 },
  'Miami, FL':                { city: 'Miami, FL',       usgs: '02288990', lat: 25.7617,  lon: -80.1918,  threshold: 5.0  },
  'Nashville, TN':            { city: 'Nashville, TN',   usgs: '03431500', lat: 36.1627,  lon: -86.7816,  threshold: 40.0 },
  'Sacramento, CA':           { city: 'Sacramento, CA',  usgs: '11447650', lat: 38.5816,  lon: -121.4944, threshold: 25.0 },
}

export const WMO_CODES: Record<number, string> = {
  0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
  45: 'Foggy', 48: 'Icy fog',
  51: 'Light drizzle', 53: 'Moderate drizzle',
  61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
  71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
  80: 'Rain showers', 81: 'Moderate showers', 82: 'Violent showers',
  95: 'Thunderstorm', 96: 'Thunderstorm + hail', 99: 'Thunderstorm + heavy hail',
}

export const SEVERITY_COLORS: Record<Severity, { bg: string; border: string; text: string; badge: string }> = {
  CRITICAL: { bg: 'bg-red-950/60',    border: 'border-red-500',    text: 'text-red-400',    badge: 'bg-red-500/20 text-red-300 border-red-500/40' },
  DANGER:   { bg: 'bg-orange-950/60', border: 'border-orange-500', text: 'text-orange-400', badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40' },
  HIGH:     { bg: 'bg-orange-950/60', border: 'border-orange-500', text: 'text-orange-400', badge: 'bg-orange-500/20 text-orange-300 border-orange-500/40' },
  WARNING:  { bg: 'bg-amber-950/60',  border: 'border-amber-500',  text: 'text-amber-400',  badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
  MEDIUM:   { bg: 'bg-amber-950/60',  border: 'border-amber-500',  text: 'text-amber-400',  badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
  LOW:      { bg: 'bg-emerald-950/60',border: 'border-emerald-500',text: 'text-emerald-400',badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
  NORMAL:   { bg: 'bg-emerald-950/60',border: 'border-emerald-500',text: 'text-emerald-400',badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
}

export const QUICK_QUERIES = [
  'Should we evacuate near the river?',
  'What is the current flood risk?',
  'Are there any recent earthquakes nearby?',
  'What does FEMA say about flash flood evacuation?',
  'What are people reporting from the field?',
  'Generate a full situation assessment report',
]

export const DEFAULT_THREAD_ID = 'dashboard-main'
