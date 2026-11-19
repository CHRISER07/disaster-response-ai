with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(355, 395):
    print(f'{i+1}: {lines[i]}', end='')
