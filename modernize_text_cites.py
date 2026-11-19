with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Introduction: replace \cite{endsley,comes,adams} with \cite{endsley,aboualola2023}
old_sa = r'a verified, spatio-temporally bounded picture of hazard severity and population vulnerability \cite{endsley,comes,adams}.'
new_sa = r'a verified, spatio-temporally bounded picture of hazard severity and population vulnerability \cite{endsley,aboualola2023}.'
assert old_sa in text, 'SA cite not found'
text = text.replace(old_sa, new_sa)

# 2. Related work: replace \cite{crisismmd,ofli,abavisani} with \cite{crisismmd,rezk2023}
old_multi = r'multimodal fusion of aerial imagery and crowdsourced microblogs \cite{crisismmd,ofli,abavisani}.'
new_multi = r'multimodal fusion of aerial imagery and crowdsourced microblogs \cite{crisismmd,rezk2023}.'
assert old_multi in text, 'Multi cite not found'
text = text.replace(old_multi, new_multi)

# 3. Related work: replace \cite{ray,aid} with \cite{manuel2024}
old_iot = r'furnish authoritative ground truth \cite{ray,aid}.'
new_iot = r'furnish authoritative ground truth \cite{manuel2024}.'
assert old_iot in text, 'IoT cite not found'
text = text.replace(old_iot, new_iot)

# 4. Related work: replace \cite{campos2020} with \cite{piryani2023}
old_temp = r'Temporal information retrieval surveys \cite{campos2020},'
new_temp = r'Temporal question answering surveys \cite{piryani2023},'
assert old_temp in text, 'Temporal cite not found'
text = text.replace(old_temp, new_temp)

# Remove any remaining ofli citations if needed
# Check if ofli is cited elsewhere:
# Line: "and official doctrine \cite{crisismmd,ofli}."
old_ofli_intro = r'and official doctrine \cite{crisismmd,ofli}.'
new_ofli_intro = r'and official doctrine \cite{crisismmd,aboualola2023}.'
if old_ofli_intro in text:
    text = text.replace(old_ofli_intro, new_ofli_intro)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Text citations modernized!')
