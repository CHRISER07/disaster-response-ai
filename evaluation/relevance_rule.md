# Evaluation Relevance Rule

A retrieved chunk $c \in \mathcal{C}$ is defined as **Relevant** ($R(c, q) = 1$) for a disaster query $q$ if and only if it satisfies both of the following objective criteria:

1. **Semantic Grounding Keyword Overlap**:
   The chunk text must contain at least one of the gold `expected_keywords` specified for query $q$ in `golden_dataset_v2.json`.

2. **Modality & Authority Alignment**:
   The chunk metadata must align with the target modality required by the query:
   - For protocol queries (`category == "protocol"`): The source must originate from official emergency guidance (`[KB - FEMA/CDC Protocol]`, FEMA ICS, USAR field guides, or CDC guidelines).
   - For eyewitness / situational awareness queries: The source must originate from field reports (`CrisisLex_T26`, social microblogs, or drone imagery descriptions).
   - For real-time sensor queries: The telemetry must originate from hydrologic or atmospheric monitoring feeds (`USGS NWIS`, `GloFAS`, `Open-Meteo`).

### Retrieval Metric Formulations:
- **Precision@7 (P@7)**: Fraction of the top 7 retrieved chunks that are relevant:
  $$\text{P@7} = \frac{1}{7} \sum_{i=1}^7 R(c_i, q)$$
- **Mean Reciprocal Rank (MRR)**: Reciprocal rank of the first relevant chunk:
  $$\text{MRR} = \frac{1}{\min \{i \mid R(c_i, q) = 1\}}$$
- **NDCG@7**: Normalized Discounted Cumulative Gain over the top 7 positions:
  $$\text{DCG@7} = \sum_{i=1}^7 \frac{R(c_i, q)}{\log_2(i + 1)}, \quad \text{NDCG@7} = \frac{\text{DCG@7}}{\text{IDCG@7}}$$

