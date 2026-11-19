import re
import numpy as np

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Strip LaTeX commands and math
clean_text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', ' ', text)
clean_text = re.sub(r'\$[^$]*\$', ' ', clean_text)
clean_text = re.sub(r'%.*', '', clean_text)
clean_text = re.sub(r'\s+', ' ', clean_text)

# Split into sentences
sentences = [s.strip() for s in re.split(r'[.!?]', clean_text) if len(s.strip().split()) > 3]
lengths = [len(s.split()) for s in sentences]

mean_len = np.mean(lengths)
std_len = np.std(lengths)
burstiness = std_len / mean_len

print(f"Total analyzed sentences: {len(sentences)}")
print(f"Mean sentence length: {mean_len:.2f} words")
print(f"Standard deviation (variance): {std_len:.2f}")
print(f"Burstiness Index (std/mean): {burstiness:.2f}")
print(f"Min length: {min(lengths)} words | Max length: {max(lengths)} words")

if burstiness > 0.45:
    print("\nResult: HIGH BURSTINESS (Typical of human academic writing; strong resistance to AI detection).")
else:
    print("\nResult: LOW BURSTINESS (Uniform, AI-like rhythm).")
