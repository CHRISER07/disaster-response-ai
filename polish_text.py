with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Natural opening sentence
old_intro = 'Natural and anthropogenic hazards---including flash floods, seismic events, wildland fires, and hazardous material releases---cause extensive casualties and destabilize critical infrastructure globally \\cite{undrr}.'
new_intro = 'Natural and anthropogenic disasters---including flash floods, earthquakes, wildland fires, and hazardous chemical releases---inflict severe casualties and destabilize infrastructure globally \\cite{undrr}.'
assert old_intro in text, 'Intro not found'
text = text.replace(old_intro, new_intro)

# 2. Safety bounds item
old_item3 = '\\item \\textit{Absence of safety guarantees:} Probabilistic alignment (RLHF, system prompts) remains inherently non-deterministic under adversarial or high-stress phrasing, failing to guarantee zero emission of lethal advice.'
new_item3 = '\\item \\textit{Absence of deterministic safety bounds:} Stochastic model alignment (RLHF, system prompt guards) remains non-deterministic under adversarial stress, failing to guarantee zero emission of hazardous directives.'
assert old_item3 in text, 'Item 3 not found'
text = text.replace(old_item3, new_item3)

# 3. TGI contribution item
old_tgi_contrib = '\\item \\textit{Temporal Grounding Injection (TGI):} Bitwise Snowflake timestamp recovery \\cite{olteanu2015}, rolling recency tagging ($\\langle\\mathsf{HIST}\\rangle$, $\\langle\\mathsf{LIVE}\\rangle$, $\\langle\\mathsf{KB}\\rangle$), and spatio-temporal query conditioning that formalizes non-stationary environmental states.'
new_tgi_contrib = '\\item \\textit{Temporal Grounding Injection (TGI):} Bitwise 64-bit Snowflake timestamp recovery, rolling recency classification ($\\langle\\mathsf{HIST}\\rangle$, $\\langle\\mathsf{LIVE}\\rangle$, $\\langle\\mathsf{KB}\\rangle$), and spatio-temporal query conditioning for non-stationary environments.'
assert old_tgi_contrib in text, 'TGI contrib not found'
text = text.replace(old_tgi_contrib, new_tgi_contrib)

# 4. Dense retrieval line
old_dense = 'Textual chunks are embedded using \\texttt{BAAI/}\\allowbreak\\texttt{bge-small-en-v1.5} \\cite{bge}. The [CLS] vector representation is $\\ell_2$-normalized:'
new_dense = 'Passage embeddings are generated using the dense encoder \\texttt{bge-small-en-v1.5} \\cite{bge}. For chunk $s$, the normalized [CLS] vector is:'
assert old_dense in text, 'Dense line not found'
text = text.replace(old_dense, new_dense)

# 5. Tool registry items
old_tools = '''\\begin{enumerate}
    \\item \\textit{Weather Feeds} (\\texttt{fetch\\_live\\_weather}): Queries Open-Meteo API for real-time temperature, precipitation rate, and gusts.
    \\item \\textit{River Gauging} (\\texttt{query\\_water\\_sensor}): Interfaces with USGS NWIS to retrieve instantaneous river stage $h$ and flood stage $h_{\\mathrm{fs}}$.
    \\item \\textit{Seismic Network} (\\texttt{query\\_recent\\_earthquakes}): Queries USGS FDSNWS for seismic events ($M \\ge 3.0$) within a 300~km radius.
    \\item \\textit{Doctrine Search} (\\texttt{search\\_official\\_protocols}): Restricted retrieval over indexed FEMA and CDC emergency response doctrines.
    \\item \\textit{Microblog Retrieval} (\\texttt{search\\_social\\_reports}): Semantic retrieval over timestamp-tagged CrisisLex incident microblogs.
    \\item \\textit{Global Corpus Search} (\\texttt{search\\_knowledge\\_base}): Unfiltered corpus search across all historical and doctrinal records.
    \\item \\textit{Damage Vision} (\\texttt{analyze\\_}\\allowbreak\\texttt{disaster\\_}\\allowbreak\\texttt{image}): Dispatches post-disaster satellite or drone imagery to a local vision-language model for building damage classification.
    \\item \\textit{Alert Gate} (\\texttt{generate\\_}\\allowbreak\\texttt{structured\\_}\\allowbreak\\texttt{alert}): Emits policy-validated emergency notifications to the EOC dashboard.
\\end{enumerate}'''

new_tools = '''\\begin{enumerate}
    \\item \\textit{Live Weather} (\\texttt{fetch\\_live\\_weather}): Queries Open-Meteo API for ambient temperature, precipitation rate, and peak wind gusts.
    \\item \\textit{River Gauging} (\\texttt{query\\_water\\_sensor}): Interfaces with USGS NWIS to retrieve instantaneous river stage $h$ and flood stage $h_{\\mathrm{fs}}$.
    \\item \\textit{Seismic Network} (\\texttt{query\\_recent\\_earthquakes}): Queries USGS FDSNWS for seismic events ($M \\ge 3.0$) within a 300~km radius.
    \\item \\textit{Doctrine Search} (\\texttt{search\\_official\\_protocols}): Restricted retrieval over indexed FEMA and CDC emergency response doctrines.
    \\item \\textit{Microblog Retrieval} (\\texttt{search\\_social\\_reports}): Semantic retrieval over timestamp-tagged CrisisLex incident microblogs.
    \\item \\textit{Global Corpus Search} (\\texttt{search\\_knowledge\\_base}): Unfiltered corpus search across all historical and doctrinal records.
    \\item \\textit{Damage Vision} (\\texttt{analyze\\_}\\allowbreak\\texttt{disaster\\_}\\allowbreak\\texttt{image}): Dispatches post-disaster satellite or drone imagery to a local vision-language model for building damage classification.
    \\item \\textit{Alert Dispatch} (\\texttt{send\\_disaster\\_alert}): Publishes policy-validated emergency notifications to the EOC dashboard.
\\end{enumerate}'''

assert old_tools in text, 'Tools not found'
text = text.replace(old_tools, new_tools)

# 6. Synthetic heading
old_synth_head = '\\textit{4) Synthetic Augmentation and Quality Verification Pipeline:}'
new_synth_head = '\\textit{4) Synthetic Pipeline and Quality Verification:}'
assert old_synth_head in text, 'Synth head not found'
text = text.replace(old_synth_head, new_synth_head)

# 7. Reference 31
old_crag = '\\bibitem{crag} S.-Q. Yan \\textit{et al.}, ``Corrective retrieval augmented generation,\'\' arXiv:2401.15884, 2024.'
new_crag = '\\bibitem{crag} S.-Q. Yan \\textit{et al.}, ``Corrective retrieval augmented generation,\'\' \\textit{arXiv preprint arXiv:2401.15884}, 2024.'
assert old_crag in text, 'CRAG ref not found'
text = text.replace(old_crag, new_crag)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: Text polish applied!')
