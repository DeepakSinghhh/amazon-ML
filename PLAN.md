# Amazon ML Challenge 2026: Business Entity Resolution Master Plan

## 0. What decides the result

| Factor | What it means for us |
|---|---|
| **Macro F0.5 per S1 entity** | Each S1 entity counts equally. A chain with 30 matches and a singleton carry the same weight. |
| **Singletons score 1.0 only if the list is empty** | One false match on a singleton costs the full 1.0. Deciding "no match" per entity is a core task, not an afterthought. |
| **Precision is weighted 2x** | When unsure, drop the match. Tune the decision rule for expected F0.5, not for F1 and not with one global threshold. |
| **Small candidate sets are ranked higher** | Blocking has to be tight (target **≤ 5 candidates per S1 on average**) and still keep recall at about 99% or more. |
| **France appears only in the test set** | Nothing can depend on US or India specifics. Validate by leaving one country out. Use multilingual models and language-agnostic features. |
| **≤ 8B parameters, MIT/Apache 2.0 license, no external lookups** | Allowed models: multilingual-e5 (MIT), bge-m3 (MIT), XLM-R and mDeBERTa-v3 (MIT), Qwen2.5 (Apache-2.0, 0.5B–7B), Mistral-7B (Apache). **Not allowed:** Llama, Gemma, geocoding APIs, business registries. |

## 1. Architecture

```
          ┌────────── S1 (reference) ──────────┐     ┌──── S2 ∪ S3 ────┐
          │                                    │     │                 │
  [P1] Normalization: unicode/accents, casing, abbreviation expansion,
       legal-suffix stripping, landmark stripping, address parsing (postal, numbers, city, state)
          │                                                            │
  [P2] Multi-retriever recall pool (per country block; country treated as an open string)
       a) char 3-5gram TF-IDF on name         d) dense ANN: fine-tuned multilingual bi-encoder
       b) word TF-IDF on name+address          e) postal-code / number-token + name-token keys
       c) reverse direction (S2/S3 -> top-k S1), so every S2/S3 record has a chance
          │  union -> about 30–50 candidates per S1, recall about 99.5%+
  [P3] Blocking pruner: cheap LightGBM over ~15 fast features plus rank and score-gap features
       -> adaptive cut per S1 (keeps 0..K) -> **candidate_pairs.tsv** (target avg ≤ 5)
          │
  [P4] Matcher (runs on candidate_pairs only)
       - rich pairwise features (name, address, embeddings, rank, mutual-rank, frequency)
       - fine-tuned cross-encoder (mDeBERTa-v3 / XLM-R) score as a feature
       - GBDT ensemble (LightGBM + CatBoost + XGBoost) -> calibrated P(match)
       - optional: LoRA-tuned Qwen2.5-7B only on the uncertain band (0.2 < p < 0.8)
          │
  [P5] Global consistency: each S2/S3 record goes to ≤ 1 S1 (S1 is deduplicated),
       S2↔S3 transitive support, per-entity cluster coherence
          │
  [P6] Per-entity decision: choose the subset (possibly empty) that maximizes EXPECTED F0.5
       -> **matching_results.tsv**
```

## 2. Build phases

### Phase 0: Setup and EDA (day 1)
Everything later depends on this phase.
- Repo layout in the submission shape: `code/business_entity_resolution/src/`, `output/`, `README.md`, `requirements.txt`
- **Exact metric implementation** (`src/metric.py`): per-S1 F0.5 with the singleton rules, macro-averaged. Unit-test it against the worked example (0.714).
- Questions the EDA must answer:
  1. Share of singletons in train, split by country and by source.
  2. Distribution of matches per S1 (0, 1, 2, …), separately for S2 and S3. **Does S2 or S3 contain duplicates of each other?**
  3. **Is each S2/S3 record matched to at most one S1?** If yes, it is a strong constraint (see P5).
  4. What fraction of S2/S3 records match nothing? These are distractors.
  5. How noisy is the country field? Do matched pairs always have the same country? This decides whether hard blocking on country is safe.
  6. The noise catalogue for each source: legal suffixes, abbreviations, landmark phrases, postal formats, typos, transliteration.
  7. **Test set, unlabeled:** French record formats (SARL/SAS/EURL, "Rue/Av./Bd", 5-digit CP, accents), and the size of each source and country.
  8. Chains with the same name at different addresses. Address has to break these ties, and they are where precision is lost.
- Validation protocol (fixed from day 1):
  - 5-fold **GroupKFold on S1 entity**. Each fold keeps the whole S2/S3 universe, so the distractors stay realistic.
  - **Leave-one-country-out** (train US → validate India, and the reverse) as a proxy for France. Every design choice must also help here, not only in-country.

### Phase 1: Normalization library (days 1–2)
`src/normalize.py`: pure functions, heavily unit-tested.
- Unicode NFKD, accent folding (keep a copy of the original), lowercasing, punctuation, `&` → `and`, and `et` for French.
- Legal suffixes, as a multilingual dictionary written from general domain knowledge (not an external lookup):
  - US: inc, corp, corporation, llc, ltd, co, company, lp, llp, pllc
  - India: pvt, private, ltd, limited, llp, opc, "(p) ltd", enterprises?, traders?
  - France: sarl, sas, sasu, sa, eurl, sci, snc, "et cie"
  - Output three variants: `name_core`, `legal_form` (canonical), `name_full_norm`.
- Address abbreviation expansion: st/street, rd/road, ave/av/avenue, blvd/bd/boulevard, marg, nagar, sector, "r." → rue, pl/place, chem/chemin, fbg/faubourg, and so on.
- Landmark stripping: `near|opp|opposite|behind|beside|next to|in front of|pres de|en face de ...` followed by the phrase. Keep the landmark text as its own field.
- Component extraction with regex, not an external parser: postal code (6-digit IN, 5-digit US/FR, ZIP+4), house/building numbers, unit/floor, city (the last tokens before state/postal), state (abbreviation map).
- Phonetic keys (Double Metaphone) and a transliteration-tolerant skeleton (drop vowels, collapse doubled letters) to handle Indian name variants.
- Acronym generation (`International Business Machines` → `ibm`) and detection.

### Phase 2: Candidate generation and blocking (days 2–4)
Candidate sets are ranked separately in the final judging, so this phase gets real effort.
- **Stage A, recall pool** (`src/blocking/retrievers.py`), run per country block, with a fallback for missing or unknown country:
  - char-ngram TF-IDF cosine on `name_core`: top-k using sparse matmul with top-n pruning
  - word TF-IDF on name + address
  - key blocking on (postal code, first rare name token) and (house number, rare street token)
  - dense ANN (FAISS, HNSW or IVF-PQ at scale) over bi-encoder embeddings; see Phase 3a
  - reverse retrieval: the top-k S1 for each S2/S3 record, merged back
  - Union the sources and keep the retriever ranks and scores as features.
- **Stage B, adaptive pruner** (`src/blocking/pruner.py`): LightGBM with cheap features only (retriever scores, ranks, score gap to the best candidate, mutual rank, postal and number match). Keep a candidate if `p ≥ τ_block` **and** it is in the top-K_max. Tune τ_block for the **recall vs. average-set-size** curve and pick the knee (target ≥ 99% pair recall at ≤ 5 average candidates). Singletons will often get 0–1 candidates, which makes sets smaller still.
- Report the blocking metrics: pair completeness (the recall ceiling), reduction ratio, average and p95 candidates per S1, and the share of S1 with an empty candidate set.
- Scalability note for the write-up: everything is O(N·k) with ANN, LSH, or sharded keys. Nothing is quadratic, and all of it shards by country and postal prefix to billions of records.

### Phase 3: Models (days 3–7)
**3a. Bi-encoder for retrieval**, which needs a GPU (Colab or Kaggle T4 is fine):
- Base: `intfloat/multilingual-e5-base` (MIT), or `BAAI/bge-m3` (MIT).
- Fine-tune on the train ground-truth pairs with MultipleNegativesRankingLoss and mined hard negatives (the top TF-IDF non-matches).
- Input text: `"name: {name} | addr: {address} | country: {country}"`.
- Multilingual pretraining carries over to French without French labels.

**3b. Cross-encoder** (GPU): `microsoft/mdeberta-v3-base` (MIT) or `xlm-roberta-large` (MIT).
- Pair classification on the stage-B candidates, with hard negatives. Train it out-of-fold so that its score is a clean feature for the GBDT.
- **Synthetic self-supervised augmentation:** apply our noise functions to S1 records, including **test S1 records such as the French ones**. These are abbreviation swaps, suffix drops, typos, token reorders, and dropped components, generated from given data only. They teach invariances in French without any external data. Record this explicitly in the documentation.

**3c. Feature matcher** (CPU, `src/features/`), about 80–150 features per pair:
- Name: Levenshtein ratio, Jaro-Winkler, token_set/sort/partial ratio (rapidfuzz), Monge-Elkan, Soft-TFIDF, char-ngram TF-IDF cosine, IDF-weighted token Jaccard, acronym match, first-token match, number tokens in the name, phonetic-key match, legal-form agreement or conflict.
- Address: postal exact and prefix match, house-number set Jaccard, **number conflict** (both present and different, a strong negative), city and state match, street-token TF-IDF cosine, landmark overlap, missing-component flags.
- Embeddings: bi-encoder cosine (name, address, full) and the cross-encoder probability.
- Context and rank (usually the biggest gains): the candidate's rank among this S1's candidates, the gap to the best, **the rank of this S1 among the candidate's own top S1s (mutual-best)**, the number of S1s whose name looks like this candidate's (chain ambiguity), and name-token rarity.
- Meta: source (S2 or S3), country-equality flag. Never a one-hot of the country value.
- Model: LightGBM + CatBoost + XGBoost, trained out-of-fold, then rank-averaged or stacked with logistic regression, then **isotonic calibration** (calibrated probabilities are needed for P6).

**3d. Optional LLM verifier:** Qwen2.5-7B-Instruct (Apache-2.0, ≤ 8B), LoRA or QLoRA fine-tuned to answer "same business? yes/no".
- It only runs on the uncertain band, which keeps the cost small. Its logit is blended into the calibrated score.
- Keep it only if it improves the leave-one-country-out result.

### Phase 4: Global consistency and decisions (days 6–8)
- **One-to-one-ish assignment:** if the EDA confirms that an S2/S3 record matches at most one S1, then for each S2/S3 record keep only the S1 with the highest probability. Down-weight the others by the margin, for example with an extra "is best S1 for this record" feature, or a hard rule when the margin is large.
- **S2↔S3 support:** score S2–S3 pairs inside each S1's candidate set. When S2-x matches S1 confidently and S3-y ≈ S2-x, raise S3-y. This helps recall on multi-match entities.
- **Expected-F0.5 subset selection per S1** (`src/decide.py`):
  - Take the calibrated p_i and sort them in descending order.
  - For k = 0…n, compute E[F0.5 | predict the top-k] exactly. Sets are small, so enumeration or DP over the Poisson-binomial distribution of true positives is cheap. k = 0 scores 1.0 only if no candidate is true, with probability Π(1 − p_i).
  - Pick the k with the highest expected value. This handles singletons and the precision/recall trade-off in a principled way, and usually beats a single global threshold.
  - Add a small learned temperature or shrink on p for France, if leave-one-country-out shows over-confidence on an unseen country.

### Phase 5: Iterate on errors (days 7–10)
- Error buckets from out-of-fold predictions: false positives on singletons, chain confusions, transliteration misses, landmark-only addresses, and blocking misses.
- Fix the normalization or features for each bucket, then re-run the full out-of-fold evaluation.
- Pseudo-labeling on test (optional): add test pairs with very high confidence and mutual-best status as extra training data, mostly to adapt to France. Only keep it if the leave-one-country-out simulation shows a gain.

### Phase 6: Submission package (last 2 days, and a dry run early)
- `src/run_pipeline.py`: one command that goes data → normalize → block → features → predict → decide → both TSVs.
- Guaranteed invariants, with asserts: every test S1 row is present, no duplicates, matches ⊆ candidates, S2/S3 IDs only.
- `python3 utils/validate_submission.py ...` passes.
- Pinned `requirements.txt`, seeds fixed, cached model weights documented, and a README with exact commands.
- `Documentation_template.md` filled in: the methodology, the blocking recall vs. size curve, the feature list, ablations, license table, and a statement that no external data was used.

## 3. Validation scoreboard (maintained in `reports/`)

| Stage | Metric | Target |
|---|---|---|
| Blocking | pair recall on out-of-fold data | ≥ 99% |
| Blocking | average candidates per S1 | ≤ 5 (lower is better) |
| Matcher | pairwise AUC and log-loss | tracked |
| End-to-end | macro F0.5 on 5-fold out-of-fold data | maximize |
| Generalization | macro F0.5 with one country left out | within ~2–3 points of in-country |

## 4. Priorities, if time runs short
1. Metric, normalization, TF-IDF blocking, and a GBDT on rapidfuzz features with the expected-F0.5 decision. This is already a strong baseline; submit it early.
2. Rank and mutual-rank features, plus the 1-to-1 constraint. These usually give the largest single jump.
3. The adaptive blocking pruner, for small candidate sets.
4. Fine-tuned bi-encoder, then the cross-encoder.
5. LLM verifier and pseudo-labeling. These are the last few tenths of a point.

## 5. Compute and workflow (team: **Nexabuild**, about 15–20 days)
- **Dev container, no GPU:** EDA, normalization, blocking, features, GBDT, and the decision step. Hugging Face is not reachable from this container, so all transformer work happens on Colab.
- **Google Colab, free T4 16 GB:** runs the notebooks in `notebooks/`, which clone this repo and read the data from Google Drive.
  - Fine-tunes the bi-encoder (multilingual-e5-base or bge-m3) and the cross-encoder (mDeBERTa-v3-base). Each takes about 1–3 hours on a T4.
  - Writes model weights to Drive and pair-score / embedding files back to the repo (small parquet files).
  - Checkpoint every epoch to Drive, because free sessions disconnect.
  - Kaggle (30 GPU-hours a week) is the backup.
- **LLM verifier:** a 7B model is too slow on a free T4. Use Qwen2.5-1.5B or 3B (Apache-2.0) with LoRA, or drop it if the cross-encoder is already enough.
- **Submission schedule:** baseline by day 3–4, then about one validated improvement per submission. Always compare the local out-of-fold F0.5 with the leaderboard to make sure they move together.
