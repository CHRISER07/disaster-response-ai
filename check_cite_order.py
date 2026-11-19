import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

bib_pattern = r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)'
bibs = re.findall(bib_pattern, text[text.find(r'\begin{thebibliography}'):], re.DOTALL)

years = {}
for key, content in bibs:
    m = re.findall(r'\b(19\d\d|20\d\d)\b', content)
    yr = int(m[-1]) if m else 9999
    years[key] = yr

# Find all \cite{...}
all_cites = re.findall(r'\\cite\{([^}]+)\}', text[:text.find(r'\begin{thebibliography}')])

print("=== MULTI-KEY CITATIONS AND THEIR YEARS ===")
for c in all_cites:
    keys = [k.strip() for k in c.split(',')]
    if len(keys) > 1:
        c_years = [years.get(k, 0) for k in keys]
        is_sorted = (c_years == sorted(c_years))
        status = "CHRONOLOGICAL" if is_sorted else "OUT OF ORDER"
        print(f"\\cite{{{c}}}")
        print(f"   Years: {c_years} -> {status}")
