'use client'

import { useState, useRef, useEffect } from 'react'
import type { ChatMessage, LocationConfig } from '@/types'
import { queryAgent, streamAgent } from '@/lib/api'
import { QUICK_QUERIES, DEFAULT_THREAD_ID } from '@/lib/constants'
import { FormattedText } from '@/components/FormattedText'

interface ChatPanelProps {
  chatHistory: ChatMessage[]
  setChatHistory: React.Dispatch<React.SetStateAction<ChatMessage[]>>
  location: LocationConfig
  onClear: () => void
}

function TypingIndicator({ activeTools }: { activeTools: string[] }) {
  return (
    <div className="flex flex-col gap-1.5 px-1">
      <div className="flex items-center gap-1.5">
        <div className="flex gap-1">
          <span className="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full inline-block" />
          <span className="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full inline-block" />
          <span className="typing-dot w-1.5 h-1.5 bg-blue-400 rounded-full inline-block" />
        </div>
        <span className="text-[10px] text-slate-500">Reasoning across live data...</span>
      </div>
      {activeTools.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1">
          {activeTools.map((t, i) => (
            <span key={i} className="inline-flex items-center gap-1 px-1.5 py-0.5 bg-blue-900/30 text-blue-300 border border-blue-500/20 rounded text-[9px] mono animate-pulse">
              <span className="w-1 h-1 bg-blue-400 rounded-full" />
              Executing: {t}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ToolBadge({ tool }: { tool: string }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-[#1e2d3d] text-cyan-300 border border-cyan-500/20 rounded-full text-[9px] mono">
      <span className="w-1 h-1 bg-cyan-400 rounded-full" />
      {tool}
    </span>
  )
}

function UserBubble({ msg }: { msg: ChatMessage }) {
  return (
    <div className="chat-enter flex justify-end mb-4">
      <div className="max-w-[80%]">
        <div className="bg-blue-600/20 border border-blue-500/30 rounded-2xl rounded-tr-sm px-4 py-3">
          <p className="text-sm text-slate-200 leading-relaxed">{msg.content}</p>
        </div>
        <div className="text-[9px] text-slate-600 mt-1 text-right mono">
          {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString() : ''}
        </div>
      </div>
    </div>
  )
}

function AssistantBubble({ msg }: { msg: ChatMessage }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="chat-enter flex justify-start mb-4">
      <div className="max-w-[90%]">
        {/* AI icon */}
        <div className="flex items-center gap-2 mb-2">
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center flex-shrink-0">
            <svg className="w-2.5 h-2.5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <span className="text-[10px] text-slate-500 mono">AI ANALYST</span>
          {msg.latency_s && (
            <span className="text-[9px] text-slate-600 mono">{msg.latency_s.toFixed(1)}s</span>
          )}
          {msg.cost_usd && (
            <span className="text-[9px] text-slate-600 mono">${msg.cost_usd.toFixed(4)}</span>
          )}
        </div>

        <div className="bg-[#0f1923]/90 border border-[#1e2d3d] rounded-2xl rounded-tl-sm px-4 py-3">
          <FormattedText content={msg.content} />

          {/* Tool calls */}
          {msg.tool_calls && msg.tool_calls.length > 0 && (
            <div className="mt-3 pt-3 border-t border-[#1e2d3d]">
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors flex items-center gap-1"
              >
                <svg className={`w-3 h-3 transition-transform ${expanded ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                </svg>
                {msg.tool_calls.length} tool{msg.tool_calls.length > 1 ? 's' : ''} called
              </button>
              {expanded && (
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {msg.tool_calls.map((t, i) => <ToolBadge key={i} tool={t} />)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function ChatPanel({ chatHistory, setChatHistory, location, onClear }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [activeTools, setActiveTools] = useState<string[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, loading, streamingContent])

  const handleSubmit = async (query?: string) => {
    const text = (query ?? input).trim()
    if (!text || loading) return

    const userMsg: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }
    setChatHistory(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    setStreamingContent('')
    setActiveTools([])

    const t0 = performance.now()
    const toolsUsed: string[] = []
    let receivedContent = ''
    let doneCalled = false

    try {
      // Try SSE streaming first for low-latency interactive feedback
      await streamAgent(
        {
          query: text,
          thread_id: DEFAULT_THREAD_ID,
          location_context: location.city,
        },
        chunk => {
          // Live partial content as it arrives (may be intermediate AI thinking chunks)
          receivedContent = chunk
          setStreamingContent(chunk)
        },
        tool => {
          if (!toolsUsed.includes(tool)) {
            toolsUsed.push(tool)
          }
          setActiveTools(prev => Array.from(new Set([...prev, tool])))
        },
        err => {
          throw new Error(err)
        },
        (finalAnswer?: string) => {
          // `done` event carries the DSEL-validated final answer. Always prefer it.
          if (finalAnswer && finalAnswer.trim()) {
            receivedContent = finalAnswer
            setStreamingContent(finalAnswer)
          }
          doneCalled = true
        }
      )

      const latency = (performance.now() - t0) / 1000
      
      // If stream finished with text (from either chunk or done.final_answer), add to history
      if (receivedContent.trim()) {
        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: receivedContent,
            tool_calls: Array.from(new Set(toolsUsed)),
            latency_s: Number(latency.toFixed(2)),
            timestamp: new Date().toISOString(),
          },
        ])
      } else {
        // Fallback to standard blocking POST /agent/query if stream returned empty
        throw new Error('Stream returned empty answer')
      }
    } catch {
      // Fallback to standard blocking POST /agent/query
      try {
        const result = await queryAgent({
          query: text,
          thread_id: DEFAULT_THREAD_ID,
          location_context: location.city,
        })

        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: result.answer || 'No response from agent.',
            tool_calls: result.tool_calls,
            cost_usd: result.cost_usd,
            latency_s: result.latency_s,
            timestamp: new Date().toISOString(),
          },
        ])
      } catch (fallbackErr) {
        setChatHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: `Error: ${String(fallbackErr)}. Check that the FastAPI server is running on port 8000.`,
            timestamp: new Date().toISOString(),
          },
        ])
      }
    } finally {
      setLoading(false)
      setActiveTools([])
      setStreamingContent('')
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="flex flex-col bg-[#080c14]/40 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e2d3d] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-300 tracking-wide uppercase">AI Analyst</span>
          <span className="text-[9px] px-1.5 py-0.5 bg-blue-600/20 text-blue-400 border border-blue-500/30 rounded mono">
            Llama 3.3 / Qwen 27B
          </span>
          <span className="text-[9px] px-1.5 py-0.5 bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 rounded mono">
            SSE Stream
          </span>
        </div>
        {chatHistory.length > 0 && (
          <button
            onClick={onClear}
            className="text-[10px] text-slate-600 hover:text-red-400 transition-colors"
          >
            Clear session
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {chatHistory.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center">
            <div className="text-center mb-8">
              <div className="w-12 h-12 rounded-full bg-blue-600/10 border border-blue-500/20 flex items-center justify-center mx-auto mb-4">
                <svg className="w-6 h-6 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                </svg>
              </div>
              <div className="text-sm font-medium text-slate-300">System Ready</div>
              <div className="text-xs text-slate-500 mt-1">Monitoring {location.city}</div>
            </div>

            {/* Quick queries */}
            <div className="w-full max-w-md space-y-2">
              <div className="text-[10px] text-slate-600 uppercase tracking-widest text-center mb-3">Quick Actions</div>
              {QUICK_QUERIES.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(q)}
                  className="w-full text-left px-4 py-3 bg-[#0f1923]/80 hover:bg-[#131f2e] border border-[#1e2d3d] hover:border-blue-500/30 rounded-xl text-xs text-slate-400 hover:text-slate-200 transition-all duration-200"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {chatHistory.map((msg, i) =>
              msg.role === 'user'
                ? <UserBubble key={i} msg={msg} />
                : <AssistantBubble key={i} msg={msg} />
            )}

            {/* In-flight streaming response */}
            {loading && streamingContent && (
              <div className="chat-enter flex justify-start mb-4">
                <div className="max-w-[90%]">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
                      <svg className="w-2.5 h-2.5 text-white animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <circle cx="12" cy="12" r="10" strokeDasharray="32" strokeDashoffset="12" />
                      </svg>
                    </div>
                    <span className="text-[10px] text-blue-400 mono animate-pulse">STREAMING ANALYSIS...</span>
                  </div>
                  <div className="bg-[#0f1923]/90 border border-blue-500/40 rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg shadow-blue-900/10">
                    <FormattedText content={streamingContent} />
                  </div>
                </div>
              </div>
            )}

            {/* Loading indicator before tokens appear */}
            {loading && !streamingContent && (
              <div className="chat-enter mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-5 h-5 rounded-full bg-gradient-to-br from-blue-600 to-cyan-500 flex items-center justify-center">
                    <svg className="w-2.5 h-2.5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                    </svg>
                  </div>
                </div>
                <div className="bg-[#0f1923]/90 border border-[#1e2d3d] rounded-2xl rounded-tl-sm px-4 py-3">
                  <TypingIndicator activeTools={activeTools} />
                </div>
              </div>
            )}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 border-t border-[#1e2d3d] p-4">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about the disaster situation... (Enter to send, Shift+Enter for newline)"
            rows={2}
            disabled={loading}
            className="flex-1 bg-[#0d1117] border border-[#1e2d3d] focus:border-blue-500/60 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none transition-colors disabled:opacity-50"
          />
          <button
            onClick={() => handleSubmit()}
            disabled={loading || !input.trim()}
            className="flex-shrink-0 w-10 h-10 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/30 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-all"
          >
            <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}
