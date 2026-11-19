import re

# Old list of keys from previous run:
old_keys = [
    'undrr', 'comes', 'adams', 'endsley', 'crisismmd', 'ofli', 'singh', 'dinani',
    'olteanu2015', 'cantini', 'ji', 'yu2024', 'lewis', 'abavisani', 'humaid',
    'olteanu2014', 'yu2025', 'kim2025', 'emu3', 'ray', 'aid', 'campos2020',
    'react', 'langgraph', 'zhang2024', 'ragas', 'inan2023', 'crisisfacts',
    'freshqa', 'selfrag', 'crag', 'nemoguardrails', 'bm25', 'bge', 'rrf',
    'xview2', 'llava', 'dab200', 'zheng', 'cohen1968', 'landis1977'
]

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

bib_start = text.find(r'\begin{thebibliography}')
bib_section = text[bib_start:]

current_keys = re.findall(r'\\bibitem\{([^}]+)\}', bib_section)

print(f"Old count: {len(old_keys)}")
print(f"Current count: {len(current_keys)}")

print("\nWhat happened to the old keys?")
for k in old_keys:
    if k in current_keys:
        print(f"  [KEPT] {k}")
    else:
        print(f"  [REMOVED/REPLACED] {k}")

print("\nWhat are the new keys added?")
for k in current_keys:
    if k not in old_keys:
        print(f"  [ADDED] {k}")
