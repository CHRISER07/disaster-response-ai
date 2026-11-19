import subprocess

phrases = [
    r'\item \textit{Live Weather Telemetry} (\texttt{fetch\_live\_weather}): Queries Open-Meteo for real-time temperature, precipitation rate, and peak gusts.',
    r'\item \textit{Weather Observation Feed} (\texttt{fetch\_live\_weather}): Queries Open-Meteo for real-time temperature, precipitation, and gusts.',
    r'\item \textit{Meteorological Data} (\texttt{fetch\_live\_weather}): Queries the Open-Meteo API for ambient temperature, precipitation, and gusts.',
    r'\item \textit{Live Meteorological Feed} (\texttt{fetch\_live\_weather}): Queries Open-Meteo API for temperature, precipitation rate, and gusts.'
]

for idx, p in enumerate(phrases):
    with open('disaster_response_rag/test_tool.tex', 'w', encoding='utf-8') as f:
        f.write(rf'''\documentclass[conference]{{IEEEtran}}
\begin{{document}}
\begin{{enumerate}}
{p}
\end{{enumerate}}
\end{{document}}''')
    res = subprocess.run(['C:/Users/Acer/AppData/Local/Programs/MiKTeX/miktex/bin/x64/miktex-pdftex.exe', '-undump=pdflatex', '-interaction=nonstopmode', 'disaster_response_rag/test_tool.tex'], capture_output=True, text=True)
    with open('test_tool.log', 'r', encoding='utf-8', errors='ignore') as fl:
        log = fl.read()
    bad = [l for l in log.splitlines() if 'Underfull' in l or 'Overfull' in l]
    print(f'Phrase {idx+1}: {len(bad)} warnings -> {bad}')
