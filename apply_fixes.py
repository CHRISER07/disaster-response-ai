import re

with open('disaster_response_rag/IEEE_FINAL.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Author block
old_author = '''\\IEEEauthorblockN{
Christopher Aseer J Albert\\textsuperscript{1},
Prisha Andrew Bhosale\\textsuperscript{2},
Ashutosh Monish Bhonsle\\textsuperscript{3},
Chaitanya Vijaykumar Mahamuni\\textsuperscript{4},\\\\
Dakshayani G\\textsuperscript{5}
}'''

new_author = '''\\IEEEauthorblockN{
Christopher Aseer J Albert\\textsuperscript{1},
Prisha Andrew Bhosale\\textsuperscript{2},
Ashutosh Monish Bhonsle\\textsuperscript{3},\\\\
Chaitanya Vijaykumar Mahamuni\\textsuperscript{4}, and
Dakshayani G\\textsuperscript{5}
}'''

assert old_author in text, 'Author block not found!'
text = text.replace(old_author, new_author)

# 2. Resizebox in Figure 1
assert '\\resizebox{\\textwidth}{!}{%' in text, 'Resizebox not found!'
text = text.replace('\\resizebox{\\textwidth}{!}{%', '\\resizebox{0.995\\textwidth}{!}{%')

# 3. Equation 6 (eq:geo_gate)
old_geo = '''\\subsection{Geospatial Routing and Hydrological Catchment Gating}
To prevent cross-watershed contamination, incoming sensor feeds undergo topological spatial routing. Each observation is annotated with coordinates $(lat, lon)$ and assigned to a hydrological sub-basin via USGS HUC-10 boundaries. The routing predicate is:
\\begin{equation}
g(x, \\ell) =
\\begin{cases}
\\mathcal{Z}_{\\mathrm{direct}}, & d(x,\\ell) \\le 5\\,\\text{km},\\; \\mathrm{HUC}_{10}(x){=}\\mathrm{HUC}_{10}(\\ell),\\\\[2pt]
\\mathcal{Z}_{\\mathrm{catchment}}, & d(x,\\ell) \\le 25\\,\\text{km},\\; \\mathrm{HUC}_{10}(x){=}\\mathrm{HUC}_{10}(\\ell),\\\\[2pt]
\\mathcal{Z}_{\\mathrm{regional}}, & 25 < d(x,\\ell) \\le 60\\,\\text{km},\\; x \\in \\mathcal{D}_{\\mathrm{meteo}},\\\\[2pt]
\\mathcal{Z}_{\\mathrm{reject}}, & \\text{otherwise,}
\\end{cases}
\\label{eq:geo_gate}
\\end{equation}
where $d(\\cdot)$ is the Haversine great-circle distance. Observations in $\\mathcal{Z}_{\\mathrm{reject}}$ are discarded immediately, preventing false flooding alerts from irrelevant upstream or trans-basin sensor readings.'''

new_geo = '''\\subsection{Geospatial Routing and Hydrological Catchment Gating}
To prevent cross-watershed contamination, incoming sensor feeds undergo topological spatial routing. Each observation is annotated with coordinates and assigned to a hydrological sub-basin via USGS HUC-10 boundaries. Let $m(x,\\ell) \\equiv \\mathbf{1}[\\mathrm{HUC}_{10}(x) = \\mathrm{HUC}_{10}(\\ell)]$ denote sub-basin concordance. The routing predicate is:
\\begin{equation}
\\label{eq:geo_gate}
g(x, \\ell) =
\\begin{cases}
\\mathcal{Z}_{\\mathrm{dir}}, & d(x,\\ell) \\le 5\\,\\text{km} \\land m(x,\\ell),\\\\[2pt]
\\mathcal{Z}_{\\mathrm{catch}}, & d(x,\\ell) \\le 25\\,\\text{km} \\land m(x,\\ell),\\\\[2pt]
\\mathcal{Z}_{\\mathrm{reg}}, & 25 < d(x,\\ell) \\le 60\\,\\text{km} \\land x \\in \\mathcal{D}_{\\mathrm{meteo}},\\\\[2pt]
\\mathcal{Z}_{\\mathrm{rej}}, & \\text{otherwise,}
\\end{cases}
\\end{equation}
where $d(\\cdot)$ is Haversine distance, $\\mathcal{Z}_{\\mathrm{dir}}$ is direct local, $\\mathcal{Z}_{\\mathrm{catch}}$ is catchment corridor, $\\mathcal{Z}_{\\mathrm{reg}}$ is regional meteorology, and $\\mathcal{Z}_{\\mathrm{rej}}$ is rejected out-of-bounds. Observations in $\\mathcal{Z}_{\\mathrm{rej}}$ are discarded immediately, preventing false flooding alerts from irrelevant upstream or trans-basin sensor readings.'''

assert old_geo in text, 'Geo gate not found!'
text = text.replace(old_geo, new_geo)

# 4. Tool Registry items
old_tools = '''\\begin{enumerate}
    \\item \\texttt{fetch\\_live\\_weather}: Queries Open-Meteo API for real-time temperature, precipitation rate, and gusts.
    \\item \\texttt{query\\_water\\_sensor}: Interfaces with USGS NWIS to retrieve instantaneous river stage $h$ and flood stage $h_{\\mathrm{fs}}$.
    \\item \\texttt{query\\_recent\\_earthquakes}: Queries USGS FDSNWS for seismic events ($M \\ge 3.0$) within a 300~km radius.
    \\item \\texttt{search\\_official\\_protocols}: Restricted retrieval over indexed FEMA and CDC emergency response doctrines.
    \\item \\texttt{search\\_social\\_reports}: Semantic retrieval over timestamp-tagged CrisisLex incident microblogs.
    \\item \\texttt{search\\_knowledge\\_base}: Unfiltered corpus search across all historical and doctrinal records.
    \\item \\texttt{analyze\\_disaster\\_image}: Dispatches post-disaster satellite or drone imagery to a local vision-language model for building damage classification.
    \\item \\texttt{generate\\_structured\\_alert}: Emits policy-validated emergency notifications to the EOC dashboard.
\\end{enumerate}'''

new_tools = '''\\begin{enumerate}
    \\item \\textit{Weather Feeds} (\\texttt{fetch\\_live\\_weather}): Queries Open-Meteo API for real-time temperature, precipitation rate, and gusts.
    \\item \\textit{River Gauging} (\\texttt{query\\_water\\_sensor}): Interfaces with USGS NWIS to retrieve instantaneous river stage $h$ and flood stage $h_{\\mathrm{fs}}$.
    \\item \\textit{Seismic Network} (\\texttt{query\\_recent\\_earthquakes}): Queries USGS FDSNWS for seismic events ($M \\ge 3.0$) within a 300~km radius.
    \\item \\textit{Doctrine Search} (\\texttt{search\\_official\\_protocols}): Restricted retrieval over indexed FEMA and CDC emergency response doctrines.
    \\item \\textit{Microblog Retrieval} (\\texttt{search\\_social\\_reports}): Semantic retrieval over timestamp-tagged CrisisLex incident microblogs.
    \\item \\textit{Global Corpus Search} (\\texttt{search\\_knowledge\\_base}): Unfiltered corpus search across all historical and doctrinal records.
    \\item \\textit{Vision Damage Assessment} (\\texttt{analyze\\_disaster\\_image}): Dispatches post-disaster satellite or drone imagery to a local vision-language model for building damage classification.
    \\item \\textit{Public Alert Dispatch} (\\texttt{generate\\_structured\\_alert}): Emits policy-validated emergency notifications to the EOC dashboard.
\\end{enumerate}'''

assert old_tools in text, 'Tools list not found!'
text = text.replace(old_tools, new_tools)

# 5. Equation 8 (eq:auth)
old_auth = '''    \\item \\textit{L4 (Alert Policy Authorization Gate):} Formal authorization predicate governing automated public alerts:
    \\begin{equation}
    \\begin{aligned}
    \\mathrm{auth}(\\hat{y}) =~& \\mathbb{1}[\\mathrm{AGR}(\\hat{y}, \\mathcal{C}) \\ge \\theta] \\cdot \\mathbb{1}[c \\in \\{\\mathrm{HIGH}, \\mathrm{MED}\\}] \\\\
    &\\cdot \\mathbb{1}[(h \\ge 1.3 h_{\\mathrm{fs}}) \\lor (Q \\ge Q_{\\mathrm{surge}}) \\lor (M \\ge 6.0)].
    \\end{aligned}
    \\label{eq:auth}
    \\end{equation}
    Automated evacuation bulletins are published only when all physical and grounding criteria evaluate to true; otherwise, publication is blocked.'''

new_auth = '''    \\item \\textit{L4 (Alert Policy Authorization Gate):} Formal authorization predicate governing automated public alerts:
    \\begin{equation}
    \\label{eq:auth}
    \\begin{aligned}
    \\mathrm{auth}(\\hat{y}) =\\;& \\mathbf{1}[\\mathrm{AGR}(\\hat{y}, \\mathcal{C}) \\ge \\theta] \\cdot \\mathbf{1}[c \\in \\{\\textsc{High}, \\textsc{Med}\\}] \\\\
    & \\cdot \\mathbf{1}[h \\ge 1.3\\,h_{\\mathrm{fs}} \\lor Q \\ge Q_{\\mathrm{surge}} \\lor M \\ge 6.0].
    \\end{aligned}
    \\end{equation}
    Automated evacuation bulletins are published only when all physical and grounding criteria evaluate to true; otherwise, publication is blocked.'''

assert old_auth in text, 'Auth equation not found!'
text = text.replace(old_auth, new_auth)

# 6. DAB-200 synthetic pipeline description
old_dab = '''\\textit{3) Reference Answer and Gold Checklist Curation:} Each scenario was authored and validated against official doctrines, including FEMA NIMS response manuals, CDC guidelines, USGS flood stage specifications, and NOAA AHPS metrics. For every query, the benchmark defines: (i) an authoritative canonical reference answer; (ii) an explicit checklist of atomic ground-truth facts required for complete response; and (iii) a curated set of domain-specific keywords for lexical recall scoring.

\\textit{4) Benchmark Partitioning:} To ensure strict scientific validity, DAB-200 is partitioned as follows:'''

new_dab = '''\\textit{3) Reference Answer and Gold Checklist Curation:} Each scenario was authored and validated against official doctrines, including FEMA NIMS response manuals, CDC guidelines, USGS flood stage specifications, and NOAA AHPS metrics. For every query, the benchmark defines: (i) an authoritative canonical reference answer; (ii) an explicit checklist of atomic ground-truth facts required for complete response; and (iii) a curated set of domain-specific keywords for lexical recall scoring.

\\textit{4) Synthetic Augmentation and Quality Verification Pipeline:} To systematically evaluate failure boundaries where empirical disaster records are sparse (such as non-stationary flood crests conflicting with viral social media rumors, or subtle upstream dam releases), controlled semi-synthetic queries were synthesized. Candidate scenarios were generated by pairing authentic sensor traces with perturbed situational premises, then filtered through a three-stage quality pipeline: (a) temporal causality verification ensuring observations strictly precede query timestamps ($\\tau_{\\mathrm{obs}} \\le \\tau_{\\mathrm{eval}}$); (b) hydrological and physical consistency audits confirming telemetry adheres to conservation-of-mass constraints; and (c) dual-expert manual review by senior emergency informatics researchers to eliminate syntactic ambiguity and calibrate rubric scoring. Items failing any audit tier were discarded.

\\textit{5) Benchmark Partitioning:} To ensure strict scientific validity, DAB-200 is partitioned as follows:'''

assert old_dab in text, 'DAB-200 section not found!'
text = text.replace(old_dab, new_dab)

with open('disaster_response_rag/IEEE_FINAL.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESS: All replacements applied!')
