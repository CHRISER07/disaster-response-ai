import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix multi-key citations to chronological
text = text.replace(r'\cite{comes,adams,endsley}', r'\cite{endsley,comes,adams}')
text = text.replace(r'\cite{dinani,olteanu2015}', r'\cite{olteanu2015,dinani}')
text = text.replace(r'\cite{humaid,olteanu2014}', r'\cite{olteanu2014,humaid}')
text = text.replace(r'FreshQA \cite{freshqa}, and TREC CrisisFACTS \cite{crisisfacts}',
                    r'TREC CrisisFACTS \cite{crisisfacts}, and FreshQA \cite{freshqa}')

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Chronological ordering applied in text citations!')
