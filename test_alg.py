import subprocess

alg_test = r'''\documentclass[conference]{IEEEtran}
\usepackage{algorithm,algorithmic,amsmath,amssymb}
\begin{document}
\begin{algorithm}[!t]
\caption{Deterministic Safety Enforcement Layer (DSEL)}
\begin{algorithmic}[1]
\STATE \textbf{// Tier L2: Provenance and Citation Audit}
\STATE Verify that all factual claims match evidence IDs in $\mathcal{C}$
\IF{unanchored dynamic assertions detected}
  \STATE Inject advisory caution: $\hat{y} \leftarrow \mathrm{InjectWarning}(\hat{y})$
\ENDIF
\STATE \textbf{// Tier L3: Anchor Grounding Ratio Check}
\STATE Compute $\mathrm{AGR}(\hat{y}, \mathcal{C})$ via Eq.~(7)
\IF{$\mathrm{AGR}(\hat{y}, \mathcal{C}) < \theta$}
  \STATE Suppress directive: $y^* \leftarrow \mathrm{SafeAbstain}(q)$
  \RETURN $y^*, \mathcal{M}_{\mathrm{audit}}$
\ENDIF
\STATE \textbf{// Tier L4: Alert Policy Authorization Gate}
\IF{$\hat{y}$ contains evacuation order or public emergency bulletin}
  \STATE Compute authorization state $\mathrm{auth}(\hat{y})$ via Eq.~(8)
  \IF{$\mathrm{auth}(\hat{y}) = 0$}
    \STATE Downgrade alert to advisory status; suppress evacuation order
  \ENDIF
\ENDIF
\STATE $y^* \leftarrow \hat{y}$; append record to persistent audit log
\RETURN $y^*, \mathcal{M}_{\mathrm{audit}}$
\end{algorithmic}
\end{algorithm}
\end{document}'''

with open('disaster_response_rag/test_alg.tex', 'w', encoding='utf-8') as f:
    f.write(alg_test)

res = subprocess.run(['C:/Users/Acer/AppData/Local/Programs/MiKTeX/miktex/bin/x64/miktex-pdftex.exe', '-undump=pdflatex', '-interaction=nonstopmode', 'disaster_response_rag/test_alg.tex'], capture_output=True, text=True)
with open('test_alg.log', 'r', encoding='utf-8', errors='ignore') as fl:
    log = fl.read()

bad = [l for l in log.splitlines() if 'Underfull' in l or 'Overfull' in l]
print(f'Alg warnings: {len(bad)}')
if len(bad) == 0:
    print('ZERO WARNINGS IN ALGORITHM 2!')
