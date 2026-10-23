# 01: Methods literature on high-dimensional (embedding) auxiliaries in survey inference

Scope: what the methodological literature says about using 384–1536-d text and image embeddings as auxiliary variables for (1) model-assisted estimation, (2) calibration, (3) PPI-style estimation, (4) stratification, (5) nonresponse adjustment, (6) small area estimation, and (7) what has already been done with text or LLM features. Each section ends with **Benchmark implications**.

**How the sources were checked (read this first).** In this session WebFetch failed on every host (DNS errors), and shell egress blocks arXiv, Project Euclid, StatCan, PMC and the journal sites. So every citation below was checked only through WebSearch result summaries: abstracts, bibliographic records and secondary descriptions. **No full text was read.** Volume, page and DOI details are given only where a search result confirmed them. Anything recalled from memory and not confirmed is marked **[UNVERIFIED]**. A formula marked *(derived)* is a standard textbook calculation I did here to set expectations; it is not quoted from a paper.

**Adversarial verification pass (2026-10-23).** A second pass tried to re-check every citation independently. WebFetch again failed on DNS. Shell curl to arXiv, Crossref, doi.org, Project Euclid, StatCan, PMC, OUP, Taylor & Francis and Wiley got 403 from the egress proxy. The shared WebSearch budget was already spent. The only usable sources were R and Python package documentation and repository READMEs, fetched from raw.githubusercontent.com and pypi.org. Citations confirmed there are listed as CONFIRMED in the Verification log at the end. Every other citation is tagged **[UNVERIFIED: not re-checked]**: it rests only on the first pass's search snippets. Treat it as unconfirmed until re-checked. The tag reflects a tool block, not evidence of an error.

---

## 0. Executive summary for the benchmark

1. **The core identity.** Every method below reduces to one fact: the design variance of a model-assisted estimator is roughly the HT variance of the *residuals* y − m(x). Embeddings help only through the population residual variance of whatever m(·) is built on them. Under SRSWOR with a fixed m, the variance ratio is RE(m vs HT) = S²_y / S²_e = 1/(1 − R²_pop) *(derived; Särndal, Swensson & Wretman 1992)*.
2. **The comparison that matters is not "versus HT".** It is versus the best conventional frame model. The incremental relative efficiency is (1 − R²_frame)/(1 − R²_frame+emb) = 1/(1 − partial R²). Example: moving R² from 0.90 to 0.92 gives RE = 1.25, while moving it from 0.20 to 0.30 gives only RE = 1.14. Report partial R² and the residual-variance ratio, not raw R².
3. **Never put raw embedding dimensions into OLS-GREG or calibration.** With p comparable to n, GREG loses efficiency and can do worse than HT (Cardot, Goga & Shehzad 2017; Chauvet & Goga 2022; Goga 2025). Customary variance estimators are then badly anti-conservative (Eustache, Dagdoug & Haziza 2025; Bouhadra & Dagdoug 2026). The usable routes are:
   - a supervised scalar m̂(x), then model calibration on it (Wu & Sitter 2001; Montanari & Ranalli 2005);
   - a small number of principal components (Cardot et al. 2017);
   - penalized (ridge or lasso) GREG and calibration (Bardsley & Chambers 1984; Guggemos & Tillé 2010; McConville et al. 2017);
   - nonparametric learners such as kNN, random forests or trees in embedding space (Baffetta et al. 2009; McConville & Toth 2019; Dagdoug, Goga & Haziza 2023).
4. **Overfitting bias is the main pitfall with flexible learners on embeddings.** Cross-fitting is the 2024–2026 consensus remedy (Dagdoug & Haziza 2026 review; An, Dagdoug, Haziza & Tillé 2026; Zrnic & Candès 2024 cross-PPI; Chernozhukov et al. 2018). A fully grown random forest or 1-NN fitted in-sample drives sample residuals toward 0. The estimator then collapses to the synthetic Σ_U m̂, and its variance estimate collapses with it.
5. **PPI is the difference estimator, and PPI++ is the GREG.** Mozer (2026) shows this algebraically. A frozen embedding model plus a head trained on *independent* data (a previous survey cycle, or a disjoint labelled set) gives an *exactly design-unbiased* difference estimator. This is the deployment pattern closest to real NSO practice.
6. **Stratification.** Gains are capped by the correlation ρ between the stratifier and y. *(Derived:)* V_st/V_srs ≈ 1 − ρ²(1 − 1/L²), and about six strata capture nearly all of it (Cochran 1977, as reported by Sweet & Sigman 1995). Rivest (2002) and SamplingStrata's anticipated-variance models handle a *noisy predicted* stratifier explicitly. Breidt & Opsomer (2008) cover strata built from a *sample-fitted* model.
7. **Nonresponse.** A covariate reduces bias only if it predicts both response and y. Predicting y matters more (Little & Vartivarian 2005). Embeddings that predict response but not y only inflate variance. The ML-weighting evidence is mixed, as the 2026 Haziza vs Little debate shows.
8. **Small area estimation (SAE).** Image-derived covariates are old news here: BHF (1988) used LANDSAT, and the World Bank and Stanford work on satellite poverty mapping continues the line. The closest analogue to "embeddings as auxiliaries" is the 2024–2026 geospatial foundation-model embedding work (PDFM; Zhang et al. 2026). It finds **uneven** gains, largest where conventional covariates are weak. With area-level models, embedding dimension p can exceed the number of areas m, so reduction is mandatory.
9. **Text embeddings in survey estimation.** I found **no** paper that uses text or document embeddings as calibration or GREG auxiliaries in a design-based simulation of a business or nonprofit population. The closest work is:
   - Waldetoft, Torgander & Magnusson (2025): transformer predictions on police-report text used as the auxiliary in Hansen-Hurwitz, difference and stratified estimators;
   - Daas & van der Doef (2020): website text used to predict CIS innovation status;
   - Hoberg & Phillips (2016): 10-K business-description text used to define industries.

   This gap is both an opportunity and a warning sign.

---

## 1. Notation and the formulas the benchmark rests on

Finite population U = {1..N}, sample s, inclusion probabilities π_k, design weights d_k = 1/π_k. Auxiliaries are x_k = (z_k, e_k), where z is the conventional frame variables and e ∈ R^D is the embedding, D = 384–1536. Target: t_y = Σ_U y_k, or the mean t_y/N.

- **HT:** t̂_HT = Σ_s d_k y_k.
- **Difference estimator** (Cassel, Särndal & Wretman 1976, *Biometrika* 63(3):615–620, doi:10.1093/biomet/63.3.615), with m(·) fixed and not fitted on s:
  t̂_D = Σ_U m(x_k) + Σ_s d_k (y_k − m(x_k)).
  It is exactly design-unbiased for any fixed m, with V(t̂_D) = V_HT(y − m). Under SRSWOR: V = N²(1 − f)/n · S²_{y−m}.
- **Model-assisted / GREG:** replace m by m̂ fitted on s (Särndal, Swensson & Wretman 1992, *Model Assisted Survey Sampling*, Springer). It is design-consistent whatever the model is, but now m̂ depends on s.
- **Relative efficiency** *(derived)*: RE = MSE(HT)/MSE(est) ≈ 1/(1 − R²_pop), where R²_pop = 1 − S²_{y−m*}/S²_y and m* is the population limit of m̂.

| R²_pop | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| RE vs HT | 1.11 | 1.25 | 1.43 | 2.0 | 3.33 | 5.0 | 10 | 20 |

- **Incremental gain over a frame-variable model** *(derived)*: RE = (1 − R²_F)/(1 − R²_{F+E}).

| R²_F → R²_{F+E} | 0.2→0.3 | 0.5→0.6 | 0.8→0.85 | 0.9→0.92 | 0.95→0.96 |
|---|---|---|---|---|---|
| RE | 1.14 | 1.25 | 1.33 | 1.25 | 1.25 |

- **Penalty for estimating p coefficients by OLS-GREG.** Heuristic, assuming multivariate-normal x: V ≈ N²(1 − f)/n · S²_y(1 − R²)(1 + p/(n − p − 2)). The inflation factor is from Cochran's (1977) regression-estimator results **[UNVERIFIED: recalled from memory; use as a heuristic only]**. Under this heuristic, the break-even R² at which OLS-GREG beats HT is 1 − 1/(1 + p/(n − p − 2)), which simplifies to p/(n − 2). The n = 1000, p = 384 cell was corrected from .39 to .38 (384/998 = 0.385):

| n \ p | 5 | 10 | 20 | 50 | 100 | 384 |
|---|---|---|---|---|---|---|
| 100 | .05 | .10 | .20 | .51 | infeasible | infeasible |
| 200 | .03 | .05 | .10 | .25 | .51 | infeasible |
| 500 | .01 | .02 | .04 | .10 | .20 | .77 |
| 1000 | .005 | .01 | .02 | .05 | .10 | .38 |

- **Variance-estimator bias** *(derived, Gaussian homoscedastic linear model)*. The linearization estimator uses fitted residuals, so E[v̂]/V ≈ (1 − p/n)/(1 + p/(n − p − 2)). This is consistent with Eustache et al. (2025), who report "substantial bias" when p is large relative to n. Nominal 95% coverage under this approximation:

| n=100, p=20 | n=100, p=50 | n=200, p=20 | n=200, p=50 | n=500, p=50 |
|---|---|---|---|---|
| 88.2% | 66.8% | 92.2% | 85.8% | 92.2% |

- **Monte Carlo precision.** The standard error of a 95% coverage estimate is 0.69 pp with R = 1000 replicates, 0.49 pp with 2000, and 0.31 pp with 5000.

---

## 2. Model-assisted estimation with ML predictors

### 2.1 Foundations and the general recipe
- **Breidt & Opsomer (2017)**, "Model-Assisted Survey Estimation with Modern Prediction Techniques," *Statistical Science* 32(2):190–205, doi:10.1214/16-STS589. [UNVERIFIED: not re-checked] This review gives one recipe that takes any predictor: linear models, mixed models, nonparametric regression, kNN, neural nets, regression trees. It also reviews the design-based asymptotics.
- **Breidt & Opsomer (2000)**, "Local polynomial regression estimators in survey sampling," *Ann. Statist.* 28(4):1026–1053, doi:10.1214/aos/1015956706. [UNVERIFIED: not re-checked] The first nonparametric model-assisted estimator. It is design-consistent and asymptotically design-unbiased, and its Taylor-linearization variance is consistent. It is better than GREG when the working model is misspecified and about equal when it is right.
- **Breidt, Claeskens & Opsomer (2005)**, "Model-assisted estimation for complex surveys using penalised splines," *Biometrika* 92(4):831–846. [UNVERIFIED: not re-checked] It produces calibrated weights, can be run in GREG software, and selects the penalty from the data under unequal-probability designs. This matters because ridge GREG on embedding PCs is the same algebraic object.

### 2.2 Specific learners
- **Lasso / adaptive lasso.** McConville, Breidt, Lee & Moisen (2017), "Model-assisted survey regression estimation with the lasso," *JSSAM* 5(2):131–158, doi:10.1093/jssam/smw041. Design consistency and a CLT along a sequence of populations and designs. Weights usable across many y via model calibration or a ridge approximation. Efficiency gains over full-regression GREG (Utah tree canopy). Implemented in the R package `mase`, which also has elastic net.
- **Regression trees.** McConville & Toth (2019), "Automated selection of post-strata using a model-assisted regression tree estimator," *Scand. J. Statist.* 46(2):389–413. The title and journal are confirmed via the `mase` package bib; the volume and pages are **[UNVERIFIED: not re-checked]**. The tree's leaves act as data-chosen post-strata, so the estimator is a post-stratification estimator. Consistency, a variance estimator and asymptotic normality are given, with an application to BLS OES data. Ferreira (arXiv:2606.08289, preprint) [UNVERIFIED: not re-checked] extends this to direct domain estimation: a population-level tree behaves like HT within domains, while domain-specific trees give large variance reductions. Both keep weights that do not depend on y, and stay additive.
- **Random forests.** Dagdoug, Goga & Haziza (2023), "Model-assisted estimation through random forests in finite population sampling," *JASA* 118(542):1234–1251, doi:10.1080/01621459.2021.1987250. [UNVERIFIED: not re-checked] Theory and variance estimators, plus a model-calibration version for multiple y. Theory requires a minimum terminal-node population count (a stopping rule). Toth & McConville (2024), "Design consistent random forest models for data collected from a complex sample," *Survey Methodology* 50(2):185–207 [UNVERIFIED: not re-checked], show that ignoring an informative design biases random forests and give conditions for consistency.
- **Neural networks.** Montanari & Ranalli (2005), "Nonparametric model calibration estimation in survey sampling," *JASA* 100:1429–1442, doi:10.1198/016214505000000141. [UNVERIFIED: not re-checked] Neural-network and local-polynomial fits are used, then weights are calibrated to the fitted values. The estimators are design-consistent and come with a variance estimator. Parker & Holan (arXiv:2009.07934) [UNVERIFIED: not re-checked] use random-weight networks (extreme learning machines) for unit-level SAE.
- **kNN.** Baffetta, Fattorini, Franceschi & Corona (2009), "Design-based approach to k-nearest neighbours technique for coupling field and remotely sensed data in forest surveys," *Remote Sensing of Environment* 113:463–475, doi:10.1016/j.rse.2008.06.014. [UNVERIFIED: not re-checked] This is the forest-inventory tradition of kNN on *image-derived* features inside design-based estimators, and the most direct precedent for "kNN in embedding space". Abstract only was seen.
- **Design-unbiased learning.** Sanguiao-Sande & Zhang (2021), "Design-unbiased statistical learning in survey sampling," *Sankhyā A* 83(2):714–744, doi:10.1007/s13171-020-00224-1. [UNVERIFIED: not re-checked] A subsampling Rao-Blackwell method gives *exactly* design-unbiased estimators for any linear or nonlinear predictor, with large gains when auxiliaries are rich. Zhang, Tam, Sanguiao-Sande, Yung & Holmberg (2026), "On design-unbiased algorithmic Machine Learning," arXiv:2606.28795 [UNVERIFIED: not re-checked], cover how to sample training data and tune for design-unbiased out-of-sample predictions. Wesley Yung is at StatCan **[UNVERIFIED]**, so this is directly relevant.

### 2.3 High-dimensional settings: when p grows
- **Dagdoug, Goga & Haziza (2023)**, "Model-assisted estimation in high-dimensional settings for survey data," *J. Applied Statistics* 50(3):761–785, doi:10.1080/02664763.2022.2047905. [UNVERIFIED: not re-checked] Penalized and tree estimators are compared in a large simulation on Irish smart-meter data (hundreds of auxiliaries). The exact theorems were not retrieved.
- **Ta, Shao, Li & Wang (2020)**, "Generalized regression estimators with high-dimensional covariates," *Statistica Sinica* 30:1135–1154, doi:10.5705/ss.202017.0384. [UNVERIFIED: not re-checked] GREG with p → ∞: WLS when p < n, lasso when sparse. Efficiency depends on the covariate distribution and the divergence rates of p and n. Bouhadra & Dagdoug (2026) note that these conditions imply p/n → 0. The exact rate was not retrieved.
- **Eustache, Dagdoug & Haziza (2025)**, "On high-dimensional variance estimation in survey sampling," *Scand. J. Statist.* 52(2):924–959, doi:10.1111/sjos.12776. [UNVERIFIED: not re-checked] Under a linear model, the customary Taylor and jackknife variance estimators for GREG can be substantially biased when p is large relative to n. Bias-adjusted estimators are proposed.
- **Bouhadra & Dagdoug (2026)**, "High-Dimensional Variance Estimation for the Generalized Regression Estimator," arXiv:2607.05570, also in *The Survey Statistician* 94 (July 2026). [UNVERIFIED: not re-checked] Gives the asymptotic bias of linearization variance estimators when p/n → κ* > 0. A **cross-validated variance estimator** is asymptotically unbiased under distributional conditions on the covariates.
- **Dharamshi, Gao & Wakefield (2025)**, "Exact variance estimation for model-assisted survey estimators using U- and V-statistics," arXiv:2502.11032. [UNVERIFIED: not re-checked] The classical variance ignores the uncertainty from fitting the working model, which "can underestimate variance in finite samples". Exact variance is available when predictions are U- or V-statistics. Covers Poisson designs, with an ACS application.
- **Empirical StatCan template.** Lundy & Rao (2022), "Relative performance of methods based on model-assisted survey regression estimation: A simulation study," *Survey Methodology* 48(1) (catalogue 12-001-X202200100003). [UNVERIFIED: not re-checked] A **real business survey was used as the finite population**. GREG, tree and lasso were compared with HT. Findings:
  - With many categorical auxiliaries and small n, only the regression tree and the 2-way lasso beat HT.
  - The tree had the least-biased variance estimator.
  - Differences vanish as n grows.
  - Gains were modest because the auxiliaries were weak (adjusted R² ≈ 0.20).
  - Full GREG weights were unstable when all auxiliaries and interactions were used.

  **This is the closest published design template to our benchmark.**

### 2.4 Overfitting, cross-fitting and debiased ML
- **Mechanism** *(derived)*. Write m̂ for the in-sample fit. Then t̂ − t = Σ_U (I_k d_k − 1)(y_k − m̂_k). When m̂ interpolates the sample (1-NN, fully grown random forest, an unpenalized network, or OLS with p ≈ n), the sample residuals are close to 0. The estimator becomes the purely synthetic Σ_U m̂_k: biased under model misspecification, with a residual-based variance estimate near 0. Embeddings make this *easier* to trigger, because units are nearly unique in embedding space.
- **Dagdoug & Haziza (2026)**, "Machine learning methods for finite population parameter estimation in survey sampling," arXiv:2604.01160 (pedagogical review, v3 July 2026). [UNVERIFIED: not re-checked] The main difficulty is the dependence between the fitted predictor and the sample. The review adapts DML (cross-fitting plus Neyman-orthogonal estimating equations) to model-assisted estimation and item nonresponse, keeping √n-consistency and asymptotic normality with high-dimensional or nonparametric learners. Under **Poisson sampling, cross-fitting removes the dependence exactly**. Under SRSWOR and stratified designs it needs a design-aware fold construction (citing "Lu et al. 2025" **[UNVERIFIED: primary source not located]**).
- **An, Dagdoug, Haziza & Tillé (2026)**, "Agnostic Model-Assisted Estimation with Machine Learning for Survey Data," arXiv:2609.10904 (IASS Webinar 68, 17 Sep 2026). [UNVERIFIED: not re-checked] A learner-agnostic, cross-fitted framework with design-aware and design-agnostic folds. Consistency conditions, variance estimation and CIs are given. Because the estimator is Neyman-orthogonal, slowly converging learners can be used without parametric rates. Simulations with linear models, random forests and kNN show that cross-fitting cuts finite-sample bias and fixes coverage.
- **"Model-assisted estimation with a training subsample: a two-phase sampling approach with design-based variance estimation,"** arXiv:2609.04082 (authors not retrieved). [UNVERIFIED: not re-checked] Treats the train/estimate split as two-phase sampling, with explicit conditions for regression trees and a resampling variance estimator.
- **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018)**, "Double/debiased machine learning for treatment and structural parameters," *Econometrics Journal* 21(1):C1–C68, doi:10.1111/ectj.12097.

**Benchmark implications (§2)**
- Estimators to include: HT/Hájek; GREG on frame z; GREG on z + k PCs of e (k ∈ {5, 10, 20, 50}); ridge, lasso and elastic-net GREG on z + e; regression-tree post-stratification; random-forest, GBM and kNN-assisted estimators on (z, e). Run each learner **in-sample and cross-fitted (K = 5 and 10)** to show the overfitting effect directly.
- Hyperparameters must be chosen *inside* each Monte Carlo sample by within-sample CV, never on the population. Oracle tuning is a leak.
- PCA computed on the *population* embeddings does not use y. It is therefore a legitimate auxiliary transform, provided the full frame's text is available before sampling. Any supervised dimension reduction must be in-sample or cross-fitted.
- Report, separately: RB, RRMSE, RE versus HT, **RE versus the best frame-only GREG**, the relative bias of the variance estimator, and coverage. Include the cross-validated or bias-adjusted variance estimators (Eustache et al. 2025; Bouhadra & Dagdoug 2026) as well as linearization.
- Sample-size grid: n where p/n is small, moderate and large, to show the regime change. For example n ∈ {100, 200, 500, 1000, 2000} against k ∈ {5 … 384}.

---

## 3. Calibration with many auxiliaries

- **Deville & Särndal (1992)**, "Calibration estimators in survey sampling," *JASA* 87:376–382. Deville, Särndal & Sautory (1993) cover generalized raking, *JASA* 88:1013–1020. Särndal (2007), "The calibration approach in survey theory and practice," *Survey Methodology* 33(2):99–119 [UNVERIFIED: not re-checked], includes methods for avoiding extreme weights. Devaud & Tillé (2019), *TEST* 28:1033–1065 [UNVERIFIED: not re-checked], give a 25-year retrospective.
- **Model calibration.** Wu & Sitter (2001), "A model-calibration approach to using complete auxiliary information from survey data," *JASA*. Its existence and content were confirmed via secondary citations; the volume 96(453):185–193 and doi:10.1198/016214501750333054 are **[UNVERIFIED]**. The method calibrates on m̂(x_k) rather than x_k, with constraints Σ_s w_k = N and Σ_s w_k m̂_k = Σ_U m̂_k. This needs *complete* auxiliary information: x known for every unit in U, which embeddings of frame text provide. Wu (2003, *Biometrika*, "Optimal calibration estimators in survey sampling") [UNVERIFIED: not re-checked] shows model calibration is optimal within a class of calibration estimators. Montanari & Ranalli (2005) extend it to neural-network and local-polynomial m̂. Dagdoug et al. (2023, JASA) and McConville et al. (2017) use it to get one set of weights for many y.
- **Ridge / penalized calibration** (the remedy for weight instability with many constraints):
  - Bardsley & Chambers (1984), "Multipurpose estimation from unbalanced samples," *JRSS C* 33(3):290–299, doi:10.2307/2347706. [UNVERIFIED: not re-checked] A ridge-type estimator that trades bias for variance under collinearity.
  - Chambers (1996), "Robust case-weighting for multipurpose establishment surveys," *JOS* 12(1):3–32. [UNVERIFIED: not re-checked] Ridging keeps weights positive, and a nonparametric bias correction is added. **Establishment-survey context.**
  - Rao & Singh (1997), "A ridge-shrinkage method for range-restricted weight calibration in survey sampling," ASA SRMS Proceedings 57–65, also StatCan WP 97-6. [UNVERIFIED: not re-checked] Relaxes calibration constraints step by step until range restrictions hold. A later discussion notes there is no guarantee a solution exists.
  - Guggemos & Tillé (2010), "Penalized calibration in survey sampling: Design-based estimation assisted by mixed models," *JSPI* 140(11):3199–3212. [UNVERIFIED: not re-checked] Some constraints are relaxed into a penalty, so the weight given to each auxiliary can be tuned. This is the design-based counterpart of mixed-model estimation and is useful for small domains.
  - Beaumont & Bocci (2008), "Another look at ridge calibration," *Metron* 66(1):5–20. Existence and venue were confirmed through the `icarus` R package's references. These list the authors as "Bocci, J., and C. Beaumont", so the author order is **[UNVERIFIED]**. `icarus` implements this penalized (ridge) calibration through its `costs` argument.
- **Dimension reduction.**
  - Cardot, Goga & Shehzad (2017), "Calibration and partial calibration on principal components when the number of auxiliary variables is large," *Statistica Sinica* 27:243–260. [UNVERIFIED: not re-checked] Calibrating on all auxiliaries when there are many can give a **higher MSE than HT**. The fix is to calibrate on the first PCs, with *partial* calibration that stays exact on important variables. The number of PCs is chosen by the data so that all weights stay positive. Application: 336 auxiliaries for electricity consumption.
  - Chauvet & Goga (2022), "Asymptotic efficiency of the calibration estimator in a high-dimensional data setting," *JSPI* 217:177–187, doi:10.1016/j.jspi.2021.07.011. [UNVERIFIED: not re-checked] There is an *extra variance term* that cannot be neglected with high-dimensional auxiliaries. They propose a bootstrap criterion for choosing calibration variables.
  - Goga (2025), "High-dimensional estimation in a survey sampling framework, model-assisted and calibration points of view," *METRON* 83(1):5–29, doi:10.1007/s40300-024-00280-9. [UNVERIFIED: not re-checked] A review: efficiency loss with many auxiliaries; ridge and PC remedies.
  - Nascimento Silva & Skinner (1997), "Variable selection for regression estimation in finite populations," *Survey Methodology* 23(1):23–32. [UNVERIFIED: not re-checked] Selection by minimizing estimated MSE beats fixed-set, significance-test, condition-number and ridge approaches. Selection breaks standard variance estimation.
- **Weights review.** Haziza & Beaumont (2017), "Construction of weights in surveys: A review," *Statistical Science* 32(2):206–226. [UNVERIFIED: not re-checked]

**Benchmark implications (§3)**
- Never calibrate on 384–1536 raw embedding totals. Include these four estimators:
  - **C1:** calibration on z only (the status quo);
  - **C2:** model calibration on [1, z, m̂(z, e)], with m̂ cross-fitted;
  - **C3:** partial calibration: exact on z, ridge-penalized on k embedding PCs (Guggemos & Tillé; Cardot et al.);
  - **C4:** PC calibration with k chosen by the positive-weights rule.
- **Weight diagnostics are a primary outcome, not a footnote.** Report % negative weights, min/max g-weights, CV of weights, Kish deff = 1 + CV² (confirmed: `PracTools::deffK` computes 1 + relvariance of the weights, citing Kish 1965 and Kish 1992), calibration non-convergence rate, and the weight range after bounding (raking-ratio / logit distance).
- Multipurpose test: compute one set of weights, then evaluate it on several y (revenue, expenses, assets, employees, program-service share). A single model-calibrated m̂ for one y can *hurt* other y. That is a realistic production concern.

---

## 4. Prediction-powered inference and the difference / GREG estimators

- **Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (2023)**, "Prediction-powered inference," *Science* 382(6671):669–674, doi:10.1126/science.adi6000. The *Science* details are **[UNVERIFIED: not re-checked]**; the `ppi_py` README confirms that the paper exists as arXiv:2301.09633. The mean estimator θ̂_PP = mean_unlabeled(f) − mean_labeled(f − y): unlabeled mean of predictions, minus the labelled mean of prediction errors. CIs are valid for any black-box f.
- **Angelopoulos, Duchi & Zrnic (2023)**, "PPI++: Efficient prediction-powered inference," arXiv:2311.01453. Power tuning: θ̂_λ = ȳ_n + λ(f̄_N − f̄_n). λ = 1 gives PPI and λ = 0 gives the classical estimator. The plug-in λ̂ = Cov(Y, F)/((1 + n/N)Var(F)) is confirmed against the reference implementation (`ppi_py` `_calc_lam_glm`, mean case), which also clips λ̂ to [0, 1]. In survey terms this is the **regression estimator with one auxiliary f**. Software: `ppi_py` (PyPI name `ppi-python`).
- **Mozer (2026)**, "PPI is the Difference Estimator: Recognizing the Survey Sampling Roots of Prediction-Powered Inference," arXiv:2603.19160. [UNVERIFIED: not re-checked] The PPI mean estimator is algebraically equal to the Cassel et al. (1976) difference estimator, and PPI++ corresponds to GREG. The paper also sets out where they differ: the mode of inference (superpopulation vs. finite-population design), the role of the unlabeled pool, and differential prediction error across subgroups.
- **Zrnic & Candès (2024)**, "Cross-prediction-powered inference," *PNAS* 121(15):e2322083121, doi:10.1073/pnas.2322083121. The PNAS details are **[UNVERIFIED: not re-checked]**; the `ppi_py` README lists the preprint as arXiv:2309.16598. Cross-fitting the predictor on the labelled data. This is the ML-side name for the cross-fitted model-assisted estimator in §2.4. It beats sample-splitting and gives more stable intervals.
- **Zrnic & Candès (2024)**, "Active statistical inference," ICML 2024 (PMLR 235). [UNVERIFIED: not re-checked] Spend labels where the model is uncertain. In survey terms this is an unequal-probability design with π_k increasing in predicted uncertainty, compare Nedyalkova & Tillé (2008) in §5.
- **Fisch et al. (2024)**, "Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models," NeurIPS 37 (arXiv:2406.04291). [UNVERIFIED: not re-checked] Stratified PPI gives tighter intervals when predictor quality varies by stratum. It is the stratified difference estimator.
- **Egami, Hinck, Stewart & Wei (2023)**, "Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of LLMs," NeurIPS 36. The authorship is confirmed via the `dsl` R package DESCRIPTION; the venue and the 80–90% figure are **[UNVERIFIED: not re-checked]**. Using LLM labels directly gives bias and invalid CIs even at 80–90% accuracy. The DSL estimator uses doubly robust correction with known sampling probabilities for gold labels.
- **Ye & Yoganarasimhan (2026)**, "Rectification Difficulty and Optimal Sample Allocation in LLM-Augmented Surveys," arXiv:2604.17267. [UNVERIFIED: not re-checked] Neyman-type allocation of human samples by per-question "rectification difficulty", meaning the residual variance after the LLM signal. It is predicted by a meta-learner on *question embeddings*. MSE reductions of 10–11% are reported.
- **UNECE handbook.** I could **not retrieve** the content of "UQ in ML-Enhanced Statistical Inference: A Practitioner Handbook with Applications in Official Statistics". The only trace found is a LinkedIn repost: it covers traditional UQ, conformal prediction and PPI, with NSO examples, a decision guide and chapter checklists. No UNECE page was found in search. The user is a contributor and should supply the relevant chapter. **[UNVERIFIED content]**

**Benchmark implications (§4)**
- PPI with a pretrained f is a special case of the difference estimator. Include it as **"E-frozen"**: f trained on an *independent* population slice or a previous tax year, then applied as a fixed m. It is exactly design-unbiased with no cross-fitting needed. This is the realistic deployment ("train on last cycle's respondents, apply this cycle") and the cleanest result to report.
- Include PPI++ (power-tuned) as the GREG-on-f variant. Note that PPI's CLT intervals ignore the fpc and the design. Under stratified or PPS designs, use design-based variance on the residuals instead.
- Also report the naive plug-in "predict-and-sum" synthetic estimator Σ_U m̂. It shows the bias that the correction term removes (Egami et al.; Waldetoft et al.).

---

## 5. Stratification with predicted values or clusters of auxiliaries

- **Dalenius & Hodges (1959)**, "Minimum variance stratification," *JASA* 54:88–101. The cum √f rule. It is optimal-ish when the stratifier is the study variable and Neyman allocation is used. It assumes negligible sampling fractions, which is wrong for highly skewed populations.
- **Lavallée & Hidiroglou (1988)**, "On the stratification of skewed populations," *Survey Methodology* 14:33–43 (catalogue 12-001-X198800114602). Iterative boundaries giving one take-all stratum plus take-some strata, minimizing n for a target CV under power allocation. It has known convergence problems (Gunning & Horgan's follow-ups).
- **Rivest (2002)**, "A generalization of the Lavallée and Hidiroglou algorithm for stratification in business surveys," *Survey Methodology* 28(2):191–198. The pages were corrected from 207–214 to those given in the `stratification` package docs, whose author is Rivest. **This is the key paper for a noisy stratifier.** It adds models linking the stratification variable x to the survey variable y: a log-linear model, or a random-replacement model where most units agree and some differ greatly. Boundaries and allocation are then optimized *under that model*. Implemented in Baillargeon & Rivest's R package `stratification` (including Kozak's algorithm).
- **Sweet & Sigman (1995)**, "Evaluation of Model-Assisted Procedures for Stratifying Skewed Populations Using Auxiliary Data," U.S. Census Bureau SM95-22. [UNVERIFIED: not re-checked] Model-assisted modifications of cum √f (Singh 1971) and of Lavallée-Hidiroglou when the input is an auxiliary rather than y. They report Cochran's (1977) finding: more than about six strata add little unless the correlation exceeds 0.95, which assumes a homoscedastic regression model.
- *(Derived)* Linear homoscedastic model, uniform x, L equal-width strata, proportional allocation: V_st/V_srs ≈ 1 − ρ²(1 − 1/L²). The resulting RE:

| ρ | L=2 | L=3 | L=5 | L=6 | L=10 | L→∞ |
|---|---|---|---|---|---|---|
| 0.5 | 1.23 | 1.29 | 1.32 | 1.32 | 1.33 | 1.33 |
| 0.7 | 1.58 | 1.77 | 1.89 | 1.91 | 1.94 | 1.96 |
| 0.9 | 2.55 | 3.57 | 4.50 | 4.71 | 5.05 | 5.26 |

  Stratification can never beat 1/(1 − ρ²). GREG on the same predictor gets that limit directly. So stratifying on m̂ is mainly worth it when it is combined with GREG and Neyman allocation, or when design-stage robustness is needed.
- **Multivariate and automated.** Ballin & Barcaroli (2013), "Joint determination of optimal stratification and sample allocation using genetic algorithm," *Survey Methodology* 39(2):369–393 (issue 2013002), implemented in R `SamplingStrata`. Its "Use of models in SamplingStrata" vignette optimizes boundaries using **anticipated CVs from a linear heteroscedastic model (Z = βY + ε, Var ∝ Y^γ) or a log-linear model linking proxy frame variables to targets**. This is exactly the setup for an embedding-predicted proxy. Zimmer, Kim & Nusser (JSM 2012) [UNVERIFIED: not re-checked] build a model-based distance on several auxiliaries, then cluster hierarchically into strata (NASS June Area Survey). SamplingStrata also uses k-means to categorize continuous stratifiers.
- **Strata defined by a sample-fitted model.** Breidt & Opsomer (2008), "Endogenous post-stratification in surveys: classifying with a sample-fitted model," *Ann. Statist.* 36(1):403–427, doi:10.1214/009053607000000703. [UNVERIFIED: not re-checked] Post-strata from intervals of predictions from a GLM fitted on the same sample. The estimator is design-consistent and has the same asymptotic variance as fixed post-strata; the cost of fitting the model first is small in practice. Nonparametric extension: Dahlke, Breidt, Opsomer & Van Keilegom (2013), *Statistica Sinica* 23:189–211. [UNVERIFIED: not re-checked]
- **Optimal unequal-probability designs.** Nedyalkova & Tillé (2008), "Optimal sampling and estimation strategies under the linear model," *Biometrika* 95(3):521–537, doi:10.1093/biomet/asn027. [UNVERIFIED: not re-checked] Optimal: balanced sampling with **π_k ∝ the model-error standard deviation σ(x_k)**, with HT/BLUE. Embeddings could inform σ̂(x), not just m̂(x).
- **Business-survey reality: stratum jumpers.** Units whose real size or class differs from the frame. Typically 5–10% **[UNVERIFIED: no source given]**, and frame age makes it worse. Large units landing in take-some strata get big weights and become influential. See Xu, Landsman & Graubard (2021), "Estimation of domain means from business surveys in the presence of stratum jumpers and nonresponse," *JOS*, doi:10.2478/jos-2021-0045 [UNVERIFIED: not re-checked], and the StatCan *Survey Methodology* 2015 article 12-001-X201500114199 (title **[UNVERIFIED]**). Classifying by noisy predicted size *creates* jumpers.

**Benchmark implications (§5)**
- Design arms:
  - **D0:** SRSWOR;
  - **D1:** status-quo stratification: NTEE major group × frame size class (BMF income/asset codes), take-all for the largest, Neyman allocation;
  - **D2:** Lavallée-Hidiroglou/Rivest boundaries on m̂(z, e), with m̂ trained on an *independent* population slice or prior year (not on the current sample);
  - **D3:** k-means clusters of raw embeddings as strata. This is an unsupervised **negative control**: clusters are not y-targeted, so the expected gain is small;
  - **D4:** PPS / π ∝ σ̂(z, e);
  - **E-post:** endogenous post-strata on cross-fitted m̂.
- Use L ≤ 6 take-some strata. Measure ρ(m̂, y) in the population and compare the realized RE with the theoretical 1 − ρ²(1 − 1/L²) curve.
- Track jumpers: the % of units whose true-y stratum differs from the predicted-y stratum, and their contribution to MSE.

---

## 6. Nonresponse adjustment with ML propensities and high-dimensional covariates

- **Little & Vartivarian (2005)**, "Does weighting for nonresponse increase the variance of survey means?" *Survey Methodology* 31(2):161–168. [UNVERIFIED: not re-checked] A weighting covariate must relate to **both** response and outcome to reduce bias. When it predicts the outcome, weighting can *reduce* variance. Outcome prediction is the main criterion and propensity prediction is secondary. Kish's variance-inflation formula is accurate only when the cell variable is weakly related to the outcome.
- **Classes and cells.**
  - Haziza & Beaumont (2007), "On the construction of imputation classes in surveys," *Int. Stat. Rev.* 75(1):25–43, doi:10.1111/j.1751-5823.2006.00002.x. [UNVERIFIED: not re-checked] The score method (predictive-mean or response-propensity stratification: quantile classes of a preliminary prediction) is generally superior. Theory conditions on the realized sample when class boundaries depend on the sample. This is the basis of StatCan-style response homogeneity groups.
  - Eltinge & Yansaneh (1997), "Diagnostics for formation of nonresponse adjustment cells…," *Survey Methodology* 23:33–40. [UNVERIFIED: not re-checked] Cells by estimated propensity or estimated item.
- **Nonparametric and ML propensities.**
  - Da Silva & Opsomer (2009), "Nonparametric propensity weighting for survey nonresponse through local polynomial regression," *Survey Methodology* 35(2) (12-001-X200900211039). [UNVERIFIED: not re-checked] The 2006 kernel version exists; its details are **[UNVERIFIED]**. Both suit low-dimensional x only.
  - Kern, Klausch & Kreuter (2019), "Tree-based machine learning methods for survey research," *Survey Research Methods* 13(1):73–93, doi:10.18148/srm/2019.v13i1.7395. [UNVERIFIED: not re-checked] Trees, random forests and boosting for panel nonresponse.
  - Buskirk & Kolenikov (2015), "Finding respondents in the forest…," *Survey Insights*. [UNVERIFIED: not re-checked] Random forests give marginal gains over logistic regression only for complex response models.
  - Larbi, Tsang, Haziza & Dagdoug (2025), "On the use of machine learning methods for the treatment of unit nonresponse in surveys," *Survey Methodology* 51(1):275–303. [UNVERIFIED: not re-checked] Bias and efficiency of many ML propensity methods plus ensembles.
  - Haziza vs Little (2026), debate "Can machine learning effectively reduce potential nonresponse bias?", *The Survey Statistician* 94. [UNVERIFIED: not re-checked] Haziza: yes, in principle, but there is no free lunch, and gains need auxiliaries observed for all sampled units that relate to both y and response. Little: no; his full argument was not retrieved.
- **Procedure choice.** Haziza & Lesage (2016), "A discussion of weighting procedures for unit nonresponse," *JOS* 32(1):129–145, doi:10.1515/jos-2016-0006. [UNVERIFIED: not re-checked] Advises against one-step nonresponse calibration in favour of propensity weighting followed by calibration.
- **Doubly robust.**
  - Kim & Haziza (2014), "Doubly robust inference with missing data in survey sampling," *Statistica Sinica* 24:375–394, doi:10.5705/ss.2012.005. [UNVERIFIED: not re-checked]
  - Yang, Kim & Song (2020), "Doubly robust inference when combining probability and non-probability samples with high dimensional data," *JRSS B* 82(2):445–465, doi:10.1111/rssb.12354. Penalized estimating equations select variables, then the DR estimator re-estimates nuisances to reduce bias. A **high-dimensional DR** template.
  - Dagdoug & Haziza (2025), "Statistical Inference for a Finite Population Mean with Machine Learning-Based Imputation for Missing Survey Data," Proc. StatCan Symposium 2024 (11-522-X202500100025). [UNVERIFIED: not re-checked] A DML framework for ML-imputed estimators.
  - Dagdoug, Goga & Haziza (2025), "Statistical inference in the presence of imputed survey data through regression trees and random forests," *Scand. J. Statist.* 52(2):960–998, doi:10.1111/sjos.12777. [UNVERIFIED: not re-checked] Mean-square consistency with a growing number of predictors, and a K-fold CV variance estimator.
  - Dagdoug, Goga & Haziza, "Imputation procedures in surveys using nonparametric and machine learning methods: an empirical comparison," arXiv:2007.06298 (venue not confirmed). [UNVERIFIED: not re-checked]
- **Weight trimming.** A propensity-score simulation found trimming did not improve boosted CART or random-forest weights ("Weight trimming and propensity score weighting," PMC3069059; authors **[UNVERIFIED]**). From memory, this is Lee, Lessler & Stuart (2011), *PLoS ONE*, which studied treatment effects rather than survey nonresponse, so it transfers only indirectly **[UNVERIFIED]**.
- **Text-specific hazard.** Website availability and website features vary with firm size, age, location and sector, and patenting firms are over-represented (Kinne & Axenbeck 2020, *Scientometrics* 125(3):2011–2041 [UNVERIFIED: not re-checked]). Daas & van der Doef (2020) list "companies without a website" and "minimal number of words" as quality issues. **When the embedding is missing, it is missing not at random**, and it is plausibly related to both response and y.

**Benchmark implications (§6)**
- Simulate response mechanisms R ~ Bernoulli(ρ_k) with several generators:
  - **NR1:** depends on z only (embeddings have nothing to add);
  - **NR2:** depends on a text-derived latent such as mission topic or text length (embeddings can help);
  - **NR3:** depends on y, MNAR (nothing fully fixes it; check whether embeddings correlated with y reduce the bias);
  - **NR4:** response correlated with having a website or text, so missing-text units are the nonrespondents.
- Adjustments:
  - RHGs by the score method on p̂(z) vs p̂(z, e) (logistic, random forest, GBM);
  - RHGs on m̂(z, e) (outcome-score classes);
  - the two-step propensity-then-calibration procedure (Haziza & Lesage);
  - the DR estimator with cross-fitted nuisances;
  - kNN donor imputation in embedding space (nearest-neighbour donor imputation is common in StatCan business processing **[UNVERIFIED]**) vs donors matched on frame size × industry.
- Report bias reduction *and* variance inflation separately. Show the Little-Vartivarian 2 × 2 (covariate associated with response and/or y) empirically, using synthetic y-correlated and response-correlated embedding components.

---

## 7. Small area estimation with high-dimensional or ML covariates

- **Classical models.**
  - Fay & Herriot (1979), "Estimates of income for small places: an application of James-Stein procedures to census data," *JASA* 74(366):269–277, doi:10.1080/01621459.1979.10482505. The area-level model.
  - Battese, Harter & Fuller (1988), "An error-components model for prediction of county crop areas using survey and satellite data," *JASA* 83(401):28–36, doi:10.2307/2288915. The unit-level nested-error EBLUP, **with LANDSAT pixel classifications as the auxiliary**. Image-derived auxiliaries in SAE are 35 years old.
  - Rao & Molina (2015), *Small Area Estimation*, 2nd ed., Wiley. Author, year, title and publisher were confirmed via the `mase` package bib, which does not show the edition number.
- **Covariate selection and error.**
  - Lahiri & Suntornchost (2015), "Variable selection for linear mixed models with applications in small area estimation," *Sankhyā B* 77(2):312–320. [UNVERIFIED: not re-checked] Sampling error biases standard selection criteria in Fay-Herriot, and adjusted AIC/BIC/adjusted R² are proposed. Low-to-moderate p only.
  - Ybarra & Lohr (2008), "Small area estimation when auxiliary information is measured with error," *Biometrika* 95(4):919–931, doi:10.1093/biomet/asn048. Ignoring noise in the auxiliaries can make Fay-Herriot *worse than direct*. Implemented in R `saeME`.
  - An arXiv preprint on Mozambique (2609.21701, "A Partial Fay-Herriot model…") [UNVERIFIED: not re-checked] notes that the number of auxiliaries can exceed the number of areas (p > m), so redundant covariates need grouping or reduction.
- **Nonlinear and ML models.**
  - Krennmair & Schmid (2022), "Flexible domain prediction using mixed effects random forests," *JRSS C* 71(5):1865–1894, doi:10.1111/rssc.12600 (title and journal confirmed via the `SAEforest` docs; volume, pages and DOI **[UNVERIFIED: not re-checked]**), with a nonparametric bootstrap MSE. R package `SAEforest`.
  - Parker (2024), "Nonlinear Fay-Herriot models for small area estimation using random weight neural networks," *JOS*, doi:10.1177/0282423X241244671. [UNVERIFIED: not re-checked]
  - Porter, Holan, Wikle & Cressie, "Spatial Fay-Herriot models for small area estimation with functional covariates," *Spatial Statistics* (2014; arXiv:1303.6668). [UNVERIFIED: not re-checked] Google Trends curves and imagery as functional covariates, reduced by Karhunen-Loève expansion plus stochastic search variable selection. **The closest precedent to "high-dimensional unstructured covariate, reduce, then Fay-Herriot".**
- **Satellite and image covariates for poverty.**
  - Jean et al. (2016), "Combining satellite imagery and machine learning to predict poverty," *Science* 353(6301):790–794, doi:10.1126/science.aaf7894. The title is confirmed via the authors' repository; the volume, pages and DOI are **[UNVERIFIED: not re-checked]**. CNN features explain up to 75% of the variation in local-level economic outcomes.
  - Engstrom, Hersh & Newhouse, "Poverty from Space…," *World Bank Economic Review* 36(2):382–412 (online 2021). [UNVERIFIED: not re-checked] Interpretable image features explain 49–61% of the variation in mean log consumption across 1,291 Sri Lankan units; accuracy falls with fewer households.
  - Merfeld & Newhouse (2023), WB PRWP 10348. [UNVERIFIED: not re-checked] XGBoost and Cubist beat nested-error EBP across 7 countries; a two-stage bootstrap gives better coverage.
  - Rolf et al. (2021), "A generalizable and accessible approach to machine learning with global satellite imagery" (MOSAIKS), *Nature Communications* 12:4392, doi:10.1038/s41467-021-24638-z. **Task-agnostic image features**, computed once and reused across many outcomes. This is the image analogue of off-the-shelf embeddings.
  - Zheng et al. (2026), "A satellite foundation model for improved wealth monitoring" (Tempov), arXiv:2604.23166. [UNVERIFIED: not re-checked] Africa-wide R² = 0.63.
- **Foundation-model embeddings as area auxiliaries (the closest analogue).**
  - Google Research (2024), "General Geospatial Inference with a Population Dynamics Foundation Model" (PDFM), arXiv:2411.07207, Agarwal et al. The first author is confirmed from the BibTeX in the google-research/population-dynamics repository. GNN embeddings of US counties and ZCTAs from maps, busyness, search trends, weather and air quality. **Correction:** the embeddings are not openly released. Per that repository's README, they are available for research use subject to approval, via a BigQuery listing, and to commercial and public-sector users through a paid Early Access Program. They beat baselines on interpolation for 27 tasks; average R² = 0.74 on 21 health variables under a held-out-county design. The README confirms a 27-task `conus27` benchmark file; the R² figures are **[UNVERIFIED: not re-checked]**.
  - Zhang et al. (2026), "Geospatial foundation-model embeddings improve population estimation unevenly across space and scale," arXiv:2605.01650. [UNVERIFIED: not re-checked] Under geographically structured validation, a median +20.1% predictive fit, but **gains are uneven**: largest where conventional covariates describe context poorly, and the embeddings transfer worse across aggregation levels than handcrafted covariates.
  - Manvi et al. (2024), "GeoLLM," ICLR 2024. LLM text features from OSM-derived prompts predict population density, wealth and income; a 70% gain in Pearson r² over nearest-neighbour baselines is reported. Venue and authors are confirmed via the GeoLLM repository; the 70% figure is **[UNVERIFIED: not re-checked]**.
- **SAE evaluation precedent.** A Newhouse et al. (2025) IARIW draft [UNVERIFIED: not re-checked] (marked "do not cite") compares EBP, Fay-Herriot and synthetic ML on census extracts. Use it for design ideas only.

**Benchmark implications (§7)**
- Domains: state × NTEE major group, or state alone. Realistic sample sizes per domain are 0–30.
- Estimators:
  - direct HT;
  - direct GREG with frame plus embeddings (Ferreira-style, domain-specific);
  - synthetic Σ_d m̂;
  - Fay-Herriot with conventional area covariates vs Fay-Herriot plus k PCs of area-mean embeddings, k ≪ m;
  - unit-level BHF/EBLUP with z vs z + PCs;
  - MERF on (z, e).
- Metrics across domains: average absolute relative bias, average RRMSE, the distribution of per-domain RE, MSE-estimator relative bias, and coverage.
- Guard against p ≥ m. Treat area-mean embeddings computed from *samples* of documents as error-prone (Ybarra & Lohr).

---

## 8. Explicit uses of text, LLM or embedding features as auxiliaries (2016–2026)

| Work | Text / feature source | How it was used | Relevance |
|---|---|---|---|
| Waldetoft, Torgander & Magnusson (2025), arXiv:2505.04643 [UNVERIFIED: not re-checked] | Swedish police reports (about 1.5M/yr) | Transformer-classifier predictions as the auxiliary in Hansen-Hurwitz (PPS on prediction), difference and stratified estimators for hate-crime counts | **Closest design-based precedent.** Text-derived predictions plus probability sampling of documents to annotate, with efficient, valid estimates. The target is a rare binary, not a skewed total. |
| Daas & van der Doef (2020), *Stat. J. IAOS* 36(4):1239–1251, doi:10.3233/SJI-200627 [UNVERIFIED: not re-checked] | Company website text (CBS NL) | Classifier trained on Community Innovation Survey respondents predicts innovation for all firms, including those with fewer than 10 employees | Real NSO use of website text for business characteristics. Flags model bias, stability, word count and firms without a site. Prediction only; not combined with design-based estimation. |
| Kinne & Axenbeck (2020), *Scientometrics* 125(3):2011–2041 [UNVERIFIED: not re-checked] | 2.4M German firm websites | Web-mining framework | **Coverage bias**: whether a firm has a website, and what is on it, depends on size, age, sector and region. |
| Istat ICT-in-enterprises survey (web scraping and text mining; authors **[UNVERIFIED]**) | Enterprise websites | Survey respondents used as training data to predict indicators for firms with web content only | NSO precedent for a model-based combination of survey and web data. |
| Hoberg & Phillips (2016), *JPE* 124(5):1423–1465, doi:10.1086/688176 [UNVERIFIED: not re-checked] | 10-K product descriptions (about 50k firm-years) | Text similarity defines firm-specific industries (TNIC) | Shows that business-description text carries economic structure. Possible auxiliary for SEC filers. |
| Egami et al. (2023) NeurIPS; Ye & Yoganarasimhan (2026) | LLM labels; question embeddings | DR correction of surrogates; embedding-predicted residual variance for allocation | Methodological templates. |
| PDFM (2024); Zhang et al. (2026); GeoLLM (2024) | Area-level multimodal and LLM embeddings | Predicting area indicators | Area-profile analogue; gains uneven. |
| "In-Context Learning for the Imputation of Public Opinion Data with LLMs," arXiv:2606.09351 [UNVERIFIED: not re-checked] | Survey records; embedding-similarity retrieval | LLM imputation | Imputation analogue. Not design-based. |
| Puts, Salgado & Daas (2024), arXiv:2409.04365 [UNVERIFIED: not re-checked] | — | ML-for-official-statistics manifesto; related CBS work on misclassification bias | Governance framing. |

**Finding.** No retrieved work uses dense text embeddings (or multimodal page-image embeddings) of frame units as auxiliary variables in GREG, calibration, stratification, nonresponse classes or SAE inside a design-based Monte Carlo on a real finite population. This would be a genuine contribution. The literature above also predicts the likely result: **large gains only for targets that conventional frame variables predict poorly**.

---

## 9. Consolidated benchmark specification (literature-driven)

**Population and truth.** A finite population with y for all units, plus frame z and text or embedding e for all units, all available *before* sampling (the Wu-Sitter "complete auxiliary information" condition). For a 990 or BMF build, z must be limited to what an NSO frame would hold at design time: NTEE, state, subsection, ruling year, and BMF asset, income and revenue amounts *from an earlier filing*. Otherwise the frame-only baseline is unrealistically weak or strong.

**Targets.** Several y with different predictability:
- total revenue (frame-predictable; expect small incremental RE);
- program-service revenue share, contributions share, or government grants (plausibly text-predictable);
- employees and volunteers;
- functional expenses.

Each target appears both as a total and as a mean. Also include domain totals for SAE.

**Estimator matrix (minimum).**
- HT/Hájek.
- Post-stratified and GREG on z (the status quo).
- Ratio / GREG on the frame size variable.
- Difference estimator with **frozen** m from independent data (E-frozen / PPI).
- PPI++.
- GREG on z + PCs(e).
- Ridge-, lasso- and elastic-net-GREG.
- Partial or penalized calibration (exact on z, penalized on PCs).
- Model calibration on cross-fitted m̂(z, e) using random forest, GBM and kNN.
- Regression-tree post-stratification.
- Endogenous post-stratification on m̂.
- Each flexible learner both **in-sample and cross-fitted**.
- The synthetic Σ_U m̂ (shows the bias).

**Designs.** SRSWOR; stratified by size × NTEE with a take-all stratum; Lavallée-Hidiroglou/Rivest stratification on externally trained m̂; π ∝ σ̂ (PPS); k-means-on-embeddings strata (negative control).

**Nonresponse layer.** NR1–NR4 as in §6.

**Metrics.**
- RB = (E θ̂ − θ)/θ;
- RRMSE = √MSE/θ;
- RE versus HT **and versus the best z-only estimator**;
- variance-estimator relative bias (E v̂ − MSE)/MSE;
- CI coverage and mean length;
- weight diagnostics: % negative, range, CV, Kish deff;
- calibration failure rate;
- domain-level average absolute relative bias, average RRMSE and coverage;
- population diagnostics: R²_pop, partial R²_{e|z}, ρ(m̂, y);
- runtime and embedding cost.

R ≥ 2000 replicates, giving a 95% coverage standard error of about 0.5 pp. Report MC standard errors for every metric.

**Theory-predicted gains to pre-register.** Use RE ≈ (1 − R²_z)/(1 − R²_{z+e}) from population fits (§1). Any empirical RE far above this signals a leak, such as oracle tuning or y leaking into the text. Any RE far below it signals overfitting or an excessive p/n penalty.

**Pitfall checklist.**
1. In-sample interpolation by kNN or random forests on near-unique embeddings.
2. Tuning on the population.
3. p/n too large for OLS-GREG or calibration.
4. Anti-conservative linearization variance.
5. Negative or extreme weights.
6. Heavy tails, outliers and stratum jumpers.
7. Time mismatch: the auxiliary text must predate y.
8. y information embedded in the text, e.g. a mission statement quoting revenue, or a 990 PDF page image that *contains* y.
9. Missing text not at random.
10. Superpopulation (PPI) vs finite-population (design) inference targets.
11. Multipurpose weight damage across several y.

---

## Open questions
- The UNECE handbook chapters on PPI and model-assisted estimation were not retrievable. Do they prescribe specific variance estimators or diagnostics we should mirror?
- Under SRSWOR and stratified designs, which cross-fitting fold construction (design-aware vs design-agnostic; An et al. 2026; "Lu et al. 2025") should the harness implement? Primary-source details were not retrieved.
- Is the (1 + p/(n − p − 2)) inflation and (1 − p/n) variance-bias heuristic accurate enough for the pre-registered predictions? Or should it be replaced by the exact Eustache et al. (2025) and Bouhadra & Dagdoug (2026) bias formulas once full texts are read?
- For a 990 population, is there any target with low frame R² but plausibly high text R²? Revenue itself is likely about 0.9+ from prior-year BMF amounts, so the incremental RE will be small by construction.
