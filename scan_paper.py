import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

buzzwords = [
    'delve', 'pivotal', 'testament', 'revolutionize', 'seamlessly', 'beacon',
    'landscape', 'it is worth noting', 'it is noteworthy', 'crucial', 'fosters',
    'game-changer', 'tapestry', 'spearhead', 'holistic', 'leverage', 'leveraging',
    'underscores', 'cornerstone', 'myriad', 'paradigm shift', 'burgeoning',
    'in conclusion', 'delving', 'paramount', 'moreover', 'furthermore'
]

print('=== SCANNING FOR AI BUZZWORDS & CLICHES ===')
for w in buzzwords:
    matches = list(re.finditer(rf'\b{w}\w*\b', text, re.IGNORECASE))
    if matches:
        print(f'Found "{w}" ({len(matches)} times):')
        for m in matches[:3]:
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            print(f'  [line ~{text[:m.start()].count(chr(10))+1}]: ...{text[start:end].replace(chr(10), " ")}...')

print('\n=== CHECKING FOR REPETITIVE PARAGRAPH OPENERS ===')
paras = text.split('\n\n')
openers = {}
for p in paras:
    p_clean = p.strip()
    if p_clean.startswith('%') or p_clean.startswith('\\') or not p_clean:
        continue
    first_word = p_clean.split()[0] if p_clean.split() else ''
    openers[first_word] = openers.get(first_word, 0) + 1

for word, count in openers.items():
    if count > 2 and len(word) > 3:
        print(f'Paragraph opening word "{word}" used {count} times')
