with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('\\begin{axis}')
print(text[pos:pos+1000])
