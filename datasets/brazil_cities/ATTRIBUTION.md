# Brazilian Cities (IBGE) — attribution

**Source:** *Brazilian Cities* by Cristiana Parada (Kaggle)
**Slug:** `crisparada/brazilian-cities`
**URL:** https://www.kaggle.com/datasets/crisparada/brazilian-cities
**Underlying data:** IBGE (Brazilian Institute of Geography and Statistics) — census, SIDRA
tables and municipal GDP; Atlas Brasil (IDHM). Field-level sources are listed in
`raw/Data_Dictionary.csv`.
**License:** see the Kaggle page (public-domain IBGE statistics compiled for education/research)

This is a **second, independent dataset** (different publisher, different subject) from the Olist
e-commerce tables in `datasets/olist/`.

## Files (unmodified from the Kaggle download)

| File | Rows | Notes |
|---|---|---|
| `raw/BRAZIL_CITIES_REV2022.CSV` | 5,578 cities × 81 columns | **used by the pipeline** — 2022 revision, comma-separated, UTF-8 |
| `raw/BRAZIL_CITIES.csv` | 5,573 × 81 | original version (semicolon-separated, CRLF) — kept for reference |
| `raw/Data_Dictionary.csv` | 116 fields | field descriptions, reference year, unit, IBGE table link |

## Columns we use

| Column | Meaning | Year |
|---|---|---|
| `CITY`, `STATE` | join key (with accent normalization) | — |
| `IBGE_RES_POP` | resident population | 2010 census |
| `IDHM`, `IDHM_Renda`, `IDHM_Educacao` | municipal human development index (overall, income, education) | 2010 |
| `GDP_CAPITA` | GDP per capita (R$) | 2016 |
| `RURAL_URBAN` | IBGE urban/rural typology (Urbano, Intermediário/Rural × Adjacente/Remoto) | 2017 |
| `CAPITAL` | 1 if state capital | — |
| `LAT`, `LONG` | city coordinates → distance to São Paulo (where most Olist sellers are) | — |

Figures pre-date the Olist orders (2016–2018); they are used as slow-moving background
characteristics of the customer's city.

## Join story

Olist `customers.customer_city` is lower-case **without accents** (`sao paulo`); IBGE `CITY` is
title-case **with accents** (`São Paulo`). Both sides are normalized (strip accents → lower-case →
collapse punctuation/whitespace) and joined on **(normalized city, state)**.

Profiling result: **99.39 % of Olist customers (98,838 / 99,441)** match a city; 4,021 of 4,304
distinct Olist cities. Unmatched rows are mostly misspellings or districts (`piumhii`, `parati`,
`embu`, `bonfim paulista`) and get state-median values.

## Data quality notes (from profiling)

| Issue | Count | Handling |
|---|---|---|
| duplicate (city, state) after normalization | 4 keys | keep the row with the larger population |
| missing values coded as `0` (population, IDHM, coordinates) | 5–13 rows per column | treat `0` as missing → state median |
| `RURAL_URBAN = "Sem classificação"` (unclassified) | 5 | map to `unknown` |
