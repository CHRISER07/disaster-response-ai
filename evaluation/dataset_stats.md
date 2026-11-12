# CrisisLex T26 Dataset Analysis

**Total events:** 26  
**Total tweets across all events:** 27,933  
**'Related and informative' tweets (used for ingestion):** 16,849 (60.3%)  

---

## Per-Event Statistics

| # | Event | Type | Region | Year | Total | Informative | % |
|---|-------|------|--------|------|------:|------------:|---|
| 1 | 2012 Colorado wildfires | Wildfire | US | 2012 | 1,200 | 685 | 57.1% |
| 2 | 2012 Costa Rica earthquake | Earthquake | Latin America | 2012 | 1,412 | 742 | 52.5% |
| 3 | 2012 Guatemala earthquake | Earthquake | Latin America | 2012 | 1,050 | 754 | 71.8% |
| 4 | 2012 Italy earthquakes | Earthquake | Europe | 2012 | 1,000 | 627 | 62.7% |
| 5 | 2012 Philipinnes floods | Flood | Philippines | 2012 | 1,000 | 761 | 76.1% |
| 6 | 2012 Typhoon Pablo | Typhoon | Philippines | 2012 | 1,000 | 678 | 67.8% |
| 7 | 2012 Venezuela refinery | Chemical | Latin America | 2012 | 1,000 | 546 | 54.6% |
| 8 | 2013 Alberta floods | Flood | Canada | 2013 | 1,000 | 685 | 68.5% |
| 9 | 2013 Australia bushfire | Wildfire | Australia | 2013 | 1,199 | 704 | 58.7% |
| 10 | 2013 Bohol earthquake | Earthquake | Philippines | 2013 | 1,000 | 425 | 42.5% |
| 11 | 2013 Boston bombings | Man_Made | US | 2013 | 1,000 | 417 | 41.7% |
| 12 | 2013 Brazil nightclub fire | Fire | Latin America | 2013 | 1,000 | 442 | 44.2% |
| 13 | 2013 Colorado floods | Flood | US | 2013 | 1,000 | 768 | 76.8% |
| 14 | 2013 Glasgow helicopter crash | Aviation | Europe | 2013 | 1,100 | 578 | 52.5% |
| 15 | 2013 LA airport shootings | Man_Made | US | 2013 | 1,032 | 682 | 66.1% |
| 16 | 2013 Lac Megantic train crash | Chemical_Fire | Canada | 2013 | 1,000 | 559 | 55.9% |
| 17 | 2013 Manila floods | Flood | Philippines | 2013 | 1,000 | 628 | 62.8% |
| 18 | 2013 NY train crash | Transit | US | 2013 | 1,000 | 904 | 90.4% |
| 19 | 2013 Queensland floods | Flood | Australia | 2013 | 1,200 | 728 | 60.7% |
| 20 | 2013 Russia meteor | Meteor_Impact | Russia | 2013 | 1,442 | 642 | 44.5% |
| 21 | 2013 Sardinia floods | Flood | Europe | 2013 | 1,000 | 631 | 63.1% |
| 22 | 2013 Savar building collapse | Building_Collapse | Bangladesh | 2013 | 1,250 | 836 | 66.9% |
| 23 | 2013 Singapore haze | Air_Quality | Asia | 2013 | 1,000 | 461 | 46.1% |
| 24 | 2013 Spain train crash | Transit | Europe | 2013 | 1,000 | 729 | 72.9% |
| 25 | 2013 Typhoon Yolanda | Typhoon | Philippines | 2013 | 1,048 | 765 | 73.0% |
| 26 | 2013 West Texas explosion | Chemical | US | 2013 | 1,000 | 472 | 47.2% |
| **Total** | **26 events** | - | - | - | **27,933** | **16,849** | **60.3%** |

---

## Per Disaster-Type Summary (for Table 1 in paper)

| Disaster Type | Events | Total Tweets | Informative | % Informative |
|---|---:|---:|---:|---:|
| Air_Quality | 1 | 1,000 | 461 | 46.1% |
| Aviation | 1 | 1,100 | 578 | 52.5% |
| Building_Collapse | 1 | 1,250 | 836 | 66.9% |
| Chemical | 2 | 2,000 | 1,018 | 50.9% |
| Chemical_Fire | 1 | 1,000 | 559 | 55.9% |
| Earthquake | 4 | 4,462 | 2,548 | 57.1% |
| Fire | 1 | 1,000 | 442 | 44.2% |
| Flood | 6 | 6,200 | 4,201 | 67.8% |
| Man_Made | 2 | 2,032 | 1,099 | 54.1% |
| Meteor_Impact | 1 | 1,442 | 642 | 44.5% |
| Transit | 2 | 2,000 | 1,633 | 81.7% |
| Typhoon | 2 | 2,048 | 1,443 | 70.5% |
| Wildfire | 2 | 2,399 | 1,389 | 57.9% |

---

## Notes for IEEE Paper

- Only tweets labeled **'Related and informative'** are ingested into ChromaDB.
- Timestamps are reconstructed from Twitter Snowflake IDs for all events.
- Each document is tagged with: `event_name`, `disaster_type`, `year`, `region`.
- The `modality` tag is set to `'text'` for all social media documents.
- Total ingested: ~16,849 social media documents across 26 events.