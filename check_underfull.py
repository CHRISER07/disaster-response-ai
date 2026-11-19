with open('disaster_response_rag/IEEE_FINAL.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print('=== ALL UNDERFULL HBOXES (BADNESS > 1500) ===')
for i, l in enumerate(lines):
    if 'Underfull \\hbox' in l:
        import re
        m = re.search(r'badness (\d+)', l)
        badness = int(m.group(1)) if m else 0
        if badness > 1500:
            print(f'Line in log {i+1}: {l.strip()}')
            for k in range(i+1, min(len(lines), i+4)):
                if lines[k].strip():
                    print('   ', lines[k].strip())
