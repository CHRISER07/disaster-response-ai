import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

bib_start = text.find(r'\begin{thebibliography}')
bib_section = text[bib_start:]

bib_items = []
for m in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', bib_section, re.DOTALL):
    key = m.group(1).strip()
    body = m.group(2).strip()
    years = re.findall(r'\b(19\d\d|20\d\d)\b', body)
    yr = int(years[-1]) if years else 9999
    bib_items.append((key, yr, body))

print(f"Total references: {len(bib_items)}")
print("\nSorted by publication year (Oldest to Newest):")
for key, yr, body in sorted(bib_items, key=lambda x: x[1]):
    first_line = body.split('\n')[0]
    print(f"[{yr}] Key: {key:18} | {first_line[:75]}")
