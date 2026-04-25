# ECON5200-FINALPROJECT
# 401(k) Eligibility → Net Financial Assets — Causal Dashboard

ECON 5200 Final Project · Double Machine Learning · SIPP 1991

Interactive Streamlit dashboard for the causal effect of 401(k) eligibility on
net financial assets, estimated via Double Machine Learning (Chernozhukov et al.,
2018) on 9,915 workers from the 1991 SIPP wave.

## What's in the dashboard

Four tabs:

1. **Executive View** — KPIs, naive-vs-causal comparison with error bars,
   compounded lifetime fan chart, live counterfactual statement
2. **What-If Scenarios** — three pre-built scenarios (conservative/base/optimistic)
   plus a single-parameter sweep with confidence band
3. **Sensitivity** — tornado chart over four parameters, cross-method robustness
   (OLS / GBM-DML / RF-DML / DML-without-pira)
4. **Methodology** — DML procedure, variables, threats summary, citations

Sidebar controls (all panels update live):

- Newly eligible workers, take-up rate, admin cost (program parameters)
- Effect multiplier (0.3×–1.5× — sensitivity haircut for unobservables)
- Bad-control toggle: include / exclude `pira` from the conditioning set
  (directly addresses Threat #2 in the memo)
- Investment horizon and real return rate (lifetime projection)

## Deployment

### 1. Push to GitHub
```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initial dashboard"
git remote add origin https://github.com/<your-handle>/<repo>.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**, link your GitHub repo
3. Set **Main file path** to `app.py`
4. Click **Deploy**

The first build takes ~2–3 minutes. Submit the resulting permanent URL
(format: `https://<repo>-<hash>.streamlit.app`).

## Updating the precomputed values

If your notebook produces different values than the defaults below, edit
the constants at the top of `app.py`:

```python
CAUSAL_ATE  = 10_321   # DML ATE
CAUSAL_SE   = 1_424    # DML standard error
NAIVE_ATE   = 19_559   # Naive OLS estimate
NAIVE_SE    = 1_618    # Naive OLS SE
ROBUST_ATE  = 9_870    # RF-nuisance robustness check
ATE_NO_PIRA = 11_240   # DML excluding pira
```

The `ATE_NO_PIRA` value is referenced in the bad-control toggle and the
robustness table — to compute it, re-run your `LinearDML.fit()` block
with `CONTROLS = ['age', 'inc', 'fsize', 'educ', 'marr', 'twoearn']`
(no `pira`) and paste the resulting ATE.

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```
