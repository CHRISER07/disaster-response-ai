import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

matches = list(re.finditer(r'\bdefinition\w*\b', text, re.IGNORECASE))
print(f'Total occurrences of "definition": {len(matches)}')
for m in matches:
    start = max(0, m.start() - 30)
    end = min(len(text), m.end() + 30)
    line_no = text[:m.start()].count('\n') + 1
    print(f'  Line {line_no}: ...{text[start:end].replace(chr(10), " ")}...')
