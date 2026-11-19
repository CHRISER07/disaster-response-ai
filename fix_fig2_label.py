with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_node = r'\node[above right, font=\tiny, text=blue!70] at (axis cs:100,0) {Telemetry Blind};'
new_node = r'\node[above left, font=\tiny, text=blue!70] at (axis cs:97,0) {Telemetry Blind};'
assert old_node in text, 'Axis node not found'
text = text.replace(old_node, new_node)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Fig 2 label fixed!')
