import fitz, re

doc = fitz.open('disaster_response_rag/IEEE_FINAL.pdf')

for p in range(len(doc)):
    text = doc[p].get_text()
    cites = re.findall(r'\[\d+(?:[–,\s]+\d+)*\]', text)
    print(f"Page {p+1} citations: {cites[:8]}")
