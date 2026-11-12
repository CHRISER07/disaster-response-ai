import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Disaster Command Center | Agentic AI Situational Intelligence',
  description:
    'Real-time AI-powered disaster response system. Monitors USGS river gauges, seismic activity, live weather, and generates FEMA-protocol-aligned situational intelligence.',
  keywords: ['disaster response', 'AI', 'agentic', 'RAG', 'FEMA', 'USGS', 'emergency management'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="grid-bg min-h-screen antialiased">
        {children}
      </body>
    </html>
  )
}
