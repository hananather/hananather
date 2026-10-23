# Embeddings as auxiliary variables: a design-based benchmark

**Claim under test:** "Text embeddings of business descriptions, websites or area profiles can serve as covariates
for stratification, nonresponse adjustment, calibration and small area estimation. This is variance reduction,
not productivity."

**Approach.** Take a real finite population where every unit's survey value is known. Rebuild the information an
agency would actually hold at design time (a register from the previous year, with text descriptions). Redraw the
survey thousands of times with a production-style design. Then compare estimators that use text embeddings against
the best conventional estimators, using bias, variance, RMSE, CI coverage and equivalent-sample-size gains.

## Testbed (proof of concept)

- **Population:** IRS Form 990 full filers in New England (MA, CT, ME, NH, VT, RI) that filed for both TY2022 and TY2023. N = 20,626.
- **Frame (design time):**
  - IRS Business Master File snapshot of 2023-08-16;
  - TY2022 return: lagged revenue, expenses, assets and employees, plus mission and program text.
- **Targets (survey year):** TY2023 values of revenue, expenses, employees, salaries, government grants, program revenue, contributions, fundraising expenses, officer compensation, volunteers, and yes/no indicators.
- **Real nonresponse mechanism:** a return filed more than 300 days after fiscal year end counts as a nonrespondent.
- **National population:** about 330k units, being added for small area estimation by state.
- **Data source:** GivingTuesday 990 data lake, public bucket `s3://gt990datalake-rawdata`.

## Layout

| Path | What |
|---|---|
| `notes/benchmark-protocol.md` | The pre-registered protocol: estimands, designs, estimators, controls, hypotheses and thresholds |
| `notes/early-findings.md` | Pilot results and the bugs they exposed |
| `notes/working-log.md` | Decision log |
| `notes/research/` | Literature (methods and applied), dataset scouting and red team, embedding models and Cohere, 990 parsing spec, the skeptic's case |
| `notebooks/01_data_and_red_team.ipynb` | Population construction and data-quality / leakage checks (pass/warn/fail table) |
| `notebooks/02_benchmark_results.ipynb` | Stage-1 diagnostics, Monte Carlo results, controls, nonresponse, frozen previous-wave models |
| `src/` | `fetch990` / `parse990_min` / `parse_all` / `build_population` (data); `embed` (cached embeddings: TF-IDF-LSA, MiniLM, Cohere, random); `survey` / `frame` / `mc` / `experiments` / `frozen` / `diagnostics` (estimation) |
| `results/` | Small CSV outputs (metrics, check tables) |

The raw XML and the derived parquet files live outside the repository (`EMBAUX_DATA`, default `/tmp/claude-0/data`) and can be rebuilt with:

```bash
python3 src/fetch990.py --index-dir $D/raw990 --out $D/xml --years 2022 2023 --states MA CT ME NH VT RI
python3 src/parse_all.py --xml-dir $D/xml --manifest $D/xml/manifest_2022.csv --year 2022 --out $D/parsed/ne_2022.parquet
python3 src/parse_all.py --xml-dir $D/xml --manifest $D/xml/manifest_2023.csv --year 2023 --out $D/parsed/ne_2023.parquet
python3 src/build_population.py --prev $D/parsed/ne_2022.parquet --curr $D/parsed/ne_2023.parquet --bmf $D/raw990/bmf_2023_8_16.csv --out $D/pop/ne_pop.parquet
```

## To run the Cohere arm (needs user action)

1. Add `api.cohere.com` to the environment's allowed network domains.
2. Add `COHERE_API_KEY` as an environment secret. A production key is needed; a trial key is capped at 1,000 calls per month.
3. Run `embed.build(cache, "cohere_v4_full", "cohere", texts, "text_full", model="embed-v4.0", input_type="classification", dim=1024)`, then rerun the notebooks.

The benchmark treats Cohere as one rung of an embedding ladder (TF-IDF-LSA → MiniLM → Cohere text → Cohere multimodal). The Form 990 data contain no PDFs or images. A genuine multimodal test needs either website screenshots (URLs exist for about 50% of units) or a population with document images, such as UK Charity Commission accounts PDFs (see `notes/research/03-…`).
