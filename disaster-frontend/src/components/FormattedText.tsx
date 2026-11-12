'use client'

import React from 'react'

interface FormattedTextProps {
  content: string
}

/**
 * High-performance, zero-dependency Markdown Formatter
 * Renders bold (**text**), status badges, bullet points, numbered lists,
 * and inline citations without raw asterisks (**) cluttering the output.
 */
export function FormattedText({ content }: FormattedTextProps) {
  if (!content) return null

  // Split into lines to format headers, lists, and evidence items
  const lines = content.split('\n')

  return (
    <div className="space-y-1.5 text-sm text-slate-200 leading-relaxed">
      {lines.map((line, lineIdx) => {
        const trimmed = line.trim()
        if (!trimmed) {
          return <div key={lineIdx} className="h-2" />
        }

        // Section Headers (e.g., Status:, Evidence:, Recommendation:, Alert Generated:)
        const isHeader =
          trimmed.startsWith('Status:') ||
          trimmed.startsWith('**Status**:') ||
          trimmed.startsWith('Evidence:') ||
          trimmed.startsWith('**Evidence**:') ||
          trimmed.startsWith('Recommendation:') ||
          trimmed.startsWith('**Recommendation**:') ||
          trimmed.startsWith('Alert Generated:') ||
          trimmed.startsWith('**Alert Generated**:')

        // Parse inline bolding (**...**) into clean <strong> tags
        const renderInline = (str: string) => {
          // Regex to match **bold** or inline citations
          const parts = str.split(/(\*\*[^*]+\*\*|\[[^\]]+\])/g)
          return parts.map((part, i) => {
            if (part.startsWith('**') && part.endsWith('**')) {
              const inner = part.slice(2, -2)
              return (
                <strong key={i} className="font-semibold text-white">
                  {inner}
                </strong>
              )
            }
            if (part.startsWith('[') && part.endsWith(']')) {
              const tag = part.slice(1, -1)
              let badgeColor = 'bg-slate-800 text-slate-300 border-slate-700'
              if (tag.includes('LIVE')) {
                badgeColor = 'bg-emerald-950/70 text-emerald-300 border-emerald-500/30'
              } else if (tag.includes('HISTORICAL')) {
                badgeColor = 'bg-amber-950/70 text-amber-300 border-amber-500/30'
              } else if (tag.includes('Confidence: HIGH')) {
                badgeColor = 'bg-blue-950/70 text-blue-300 border-blue-500/30'
              } else if (tag.includes('Confidence: LOW')) {
                badgeColor = 'bg-red-950/70 text-red-300 border-red-500/30'
              }

              return (
                <span
                  key={i}
                  className={`inline-block px-1.5 py-0.5 mx-0.5 text-[10px] mono rounded border ${badgeColor}`}
                >
                  {tag}
                </span>
              )
            }
            return <React.Fragment key={i}>{part}</React.Fragment>
          })
        }

        // Format numbered recommendation lines
        if (/^\d+\.\s/.test(trimmed)) {
          const numMatch = trimmed.match(/^(\d+\.)\s*(.*)$/)
          if (numMatch) {
            return (
              <div key={lineIdx} className="flex items-start gap-2 ml-2">
                <span className="mono text-xs font-semibold text-blue-400 mt-0.5">
                  {numMatch[1]}
                </span>
                <span className="flex-1">{renderInline(numMatch[2])}</span>
              </div>
            )
          }
        }

        // Format bullet points (e.g. - Evidence item)
        if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
          return (
            <div key={lineIdx} className="flex items-start gap-2 ml-2">
              <span className="text-blue-400 mt-1">•</span>
              <span className="flex-1">{renderInline(trimmed.slice(2))}</span>
            </div>
          )
        }

        return (
          <div
            key={lineIdx}
            className={
              isHeader
                ? 'font-semibold text-slate-100 mt-2 text-[13px] tracking-wide'
                : ''
            }
          >
            {renderInline(line)}
          </div>
        )
      })}
    </div>
  )
}

