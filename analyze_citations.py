import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract bibitems and their years
bib_pattern = r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)'
bibs = re.findall(bib_pattern, text[text.find(r'\begin{thebibliography}'):], re.DOTALL)

years = {}
for key, content in bibs:
    m = re.findall(r'\b(19\d\d|20\d\d)\b', content)
    yr = int(m[-1]) if m else 9999
    years[key] = yr

# Extract all \cite{...} in order of appearance
cites = re.findall(r'\\cite\{([^}]+)\}', text[:text.find(r'\begin{thebibliography}')])
ordered_keys = []
for c in cites:
    for k in c.split(','):
        k = k.strip()
        if k and k not in ordered_keys:
            ordered_keys.append(k)

print(f"Total bibitems in bibliography: {len(bibs)}")
print(f"Total unique cited keys in text: {len(ordered_keys)}")

print("\n--- Bibliography entries in current order ---")
for i, (key, _) in enumerate(bibs):
    yr = years.get(key, 'Unknown')
    # find where it appears in text
    app_idx = ordered_keys.index(key) + 1 if key in ordered_keys else 'NOT CITED'
    print(f"[{i+1}] {key} ({yr}) -> First cited at order #{app_idx}")
