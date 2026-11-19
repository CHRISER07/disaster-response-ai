import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace \cite{cohen1968} with \cite{cohen1968, landis1977}
text = text.replace(r'\cite{cohen1968}:', r'\cite{cohen1968, landis1977}:')

# Extract all \cite{...} in text in order of appearance
bib_start = text.find(r'\begin{thebibliography}')
text_before_bib = text[:bib_start]

all_cites = re.findall(r'\\cite\{([^}]+)\}', text_before_bib)
ordered_keys = []
for c in all_cites:
    for k in c.split(','):
        k = k.strip()
        if k and k not in ordered_keys:
            ordered_keys.append(k)

print(f"Total unique cited keys in text: {len(ordered_keys)}")

# Extract bibitems
bib_section = text[bib_start:]
bib_items_dict = {}
for m in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', bib_section, re.DOTALL):
    key = m.group(1).strip()
    body = m.group(2).strip()
    bib_items_dict[key] = body

# Reconstruct thebibliography in exact order of appearance
new_bib_entries = []
for k in ordered_keys:
    if k in bib_items_dict:
        new_bib_entries.append(f"\\bibitem{{{k}}} {bib_items_dict[k]}\n")
    else:
        print(f"WARNING: Key {k} not found in bibitems!")

# Any remaining bibitems not cited in text
remaining = [k for k in bib_items_dict if k not in ordered_keys]
print(f"Remaining un-cited keys: {remaining}")
for k in remaining:
    new_bib_entries.append(f"\\bibitem{{{k}}} {bib_items_dict[k]}\n")

new_bib_body = "\n".join(new_bib_entries)
new_bib_section = f"\\def\\IEEEbibitemsep{{-0.3pt plus 0.2pt}}\n\\begin{{thebibliography}}{{00}}\n\n{new_bib_body}\n\\end{{thebibliography}}"

# Replace old thebibliography
bib_end = text.find(r'\end{thebibliography}') + len(r'\end{thebibliography}')
text = text[:bib_start] + new_bib_section + text[bib_end:]

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"SUCCESS: All {len(ordered_keys)} citations synchronized in 1-to-1 order of appearance!")
