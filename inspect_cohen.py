with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('cohen1968')
print(text[pos-100:pos+150])
