with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Tool registry item 1
old_item1 = r'\item \textit{Live Weather} (\texttt{fetch\_live\_weather}): Queries Open-Meteo API for ambient temperature, precipitation rate, and peak wind gusts.'
new_item1 = r'\item \textit{Live Weather Telemetry} (\texttt{fetch\_live\_weather}): Queries Open-Meteo for real-time temperature, precipitation rate, and peak gusts.'
assert old_item1 in text, 'Item 1 not found'
text = text.replace(old_item1, new_item1)

# 2. Algorithm 2 clean lines
old_alg2_lines = '''  \\STATE Prepend conservative qualification: $\\hat{y} \\leftarrow \\mathrm{InjectWarning}(\\hat{y})$
\\ENDIF
\\STATE \\textbf{// Tier L3: Anchor Grounding Ratio Check}
\\STATE Compute $\\mathrm{AGR}(\\hat{y}, \\mathcal{C})$ via Eq.~\\eqref{eq:agr}
\\IF{$\\mathrm{AGR}(\\hat{y}, \\mathcal{C}) < \\theta$}
  \\STATE Suppress directive; return safe abstention: $y^* \\leftarrow \\mathrm{SafeAbstain}(q)$
  \\RETURN $y^*, \\mathcal{M}_{\\mathrm{audit}}$
\\ENDIF
\\STATE \\textbf{// Tier L4: Alert Policy Authorization Gate}
\\IF{$\\hat{y}$ contains evacuation order or public emergency bulletin}
  \\STATE Compute authorization state $\\mathrm{auth}(\\hat{y})$ via Eq.~\\eqref{eq:auth}
  \\IF{$\\mathrm{auth}(\\hat{y}) = 0$}
    \\STATE Downgrade alert to advisory status; suppress evacuation order
  \\ENDIF
\\ENDIF
\\STATE $y^* \\leftarrow \\hat{y}$; write transaction to \\texttt{incident\\_audit\\_trail.jsonl}'''

new_alg2_lines = '''  \\STATE Inject advisory caution: $\\hat{y} \\leftarrow \\mathrm{InjectWarning}(\\hat{y})$
\\ENDIF
\\STATE \\textbf{// Tier L3: Anchor Grounding Ratio Check}
\\STATE Compute $\\mathrm{AGR}(\\hat{y}, \\mathcal{C})$ via Eq.~\\eqref{eq:agr}
\\IF{$\\mathrm{AGR}(\\hat{y}, \\mathcal{C}) < \\theta$}
  \\STATE Suppress directive: $y^* \\leftarrow \\mathrm{SafeAbstain}(q)$
  \\RETURN $y^*, \\mathcal{M}_{\\mathrm{audit}}$
\\ENDIF
\\STATE \\textbf{// Tier L4: Alert Policy Authorization Gate}
\\IF{$\\hat{y}$ contains evacuation order or public emergency bulletin}
  \\STATE Compute authorization state $\\mathrm{auth}(\\hat{y})$ via Eq.~\\eqref{eq:auth}
  \\IF{$\\mathrm{auth}(\\hat{y}) = 0$}
    \\STATE Downgrade alert to advisory status; suppress evacuation order
  \\ENDIF
\\ENDIF
\\STATE $y^* \\leftarrow \\hat{y}$; append record to persistent audit log'''

assert old_alg2_lines in text, 'Alg 2 lines not found'
text = text.replace(old_alg2_lines, new_alg2_lines)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Alg 2 and Tool 1 polish applied!')
