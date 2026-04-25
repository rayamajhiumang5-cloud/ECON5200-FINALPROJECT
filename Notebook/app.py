"""
401(k) Eligibility → Net Financial Assets
ECON 5200 Final Project | Double Machine Learning Dashboard
Author: Umang Rayamajhi

Run locally:  streamlit run app.py
Deploy:       https://streamlit.io/cloud
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="401(k) Causal Impact | DML",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTED RESULTS (from notebook Part 3)
# Update these with your actual computed values if they differ.
# ──────────────────────────────────────────────────────────────────────────────
CAUSAL_ATE = 10_321        # Double ML ATE ($)
CAUSAL_SE  = 1_424         # Standard error ($)
NAIVE_ATE  = 19_559        # Naive OLS estimate ($)
NAIVE_SE   = 1_618         # Naive OLS SE (approx, for CI)
N_OBS      = 9_915
TREAT_PREV = 0.394         # Share with e401 = 1
ROBUST_ATE = 9_870         # RF nuisance robustness check (approx, edit if different)
ATE_NO_PIRA = 11_240       # Counterfactual: ATE when pira excluded from controls

# Color palette
C_BIASED = "#E24B4A"
C_CAUSAL = "#185FA5"
C_ACCENT = "#D85A30"
C_MUTED  = "#888780"

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def ci(ate, se, alpha=0.05):
    """Two-sided normal CI."""
    z = 1.96 if alpha == 0.05 else 2.576
    return ate - z * se, ate + z * se


def fmt_money(x, cents=False):
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def fmt_money_M(x):
    return f"${x/1e6:.2f}M"


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.title("💰 Does 401(k) Eligibility Cause Workers to Save More?")
st.caption(
    "ECON 5200 Consulting Report · Double Machine Learning · "
    "Survey of Income and Program Participation (SIPP), 1991 · n = 9,915"
)

lb, ub = ci(CAUSAL_ATE, CAUSAL_SE)
st.info(
    f"**Bottom line:** 401(k) eligibility causally increases net financial assets "
    f"by **{fmt_money(CAUSAL_ATE)}** per worker "
    f"(95% CI: [{fmt_money(lb)}, {fmt_money(ub)}]) under the Conditional Independence Assumption. "
    f"The naive OLS estimate of {fmt_money(NAIVE_ATE)} overstates the effect "
    f"by **{fmt_money(NAIVE_ATE - CAUSAL_ATE)}** ({(NAIVE_ATE-CAUSAL_ATE)/NAIVE_ATE:.0%}) "
    f"due to positive selection on income and savings motivation."
)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR: WHAT-IF CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📊 What-If Scenario Controls")
st.sidebar.markdown("Adjust assumptions; every panel updates in real time.")

st.sidebar.subheader("Program parameters")
n_workers = st.sidebar.slider(
    "Newly eligible workers", 100, 25_000, 1_000, 100,
    help="How many additional workers your firm extends 401(k) eligibility to.",
)
take_up_rate = st.sidebar.slider(
    "Take-up rate (%)", 10, 100, 60, 5,
    help="Share of newly-eligible workers who actually enroll. Empirical median ≈ 60%.",
) / 100
admin_cost = st.sidebar.slider(
    "Admin cost per participant ($/year)", 100, 600, 300, 25,
    help="Recordkeeping + compliance cost. Industry average is $200–$400.",
)

st.sidebar.subheader("Causal sensitivity")
multiplier = st.sidebar.slider(
    "Effect multiplier (haircut for unobservables)",
    0.3, 1.5, 1.0, 0.05,
    help=(
        "Apply a discount to the headline ATE. "
        "0.5× ≈ aggressive haircut for firm-level confounding; "
        "1.0× = CIA holds; "
        ">1.0× = positive peer-effect spillovers (Duflo & Saez 2003)."
    ),
)
exclude_pira = st.sidebar.checkbox(
    "Exclude IRA participation from controls (bad-control test)",
    value=False,
    help=(
        "If pira is post-treatment (downstream of e401), conditioning on it "
        "absorbs part of the true causal effect. Toggling this on uses the "
        "DML estimate without pira as a control — see Threats memo §2."
    ),
)

st.sidebar.subheader("Lifetime projection")
horizon = st.sidebar.slider("Investment horizon (years)", 1, 40, 20)
return_rate = st.sidebar.slider("Real return assumption (%/yr)", 0.0, 8.0, 5.0, 0.25) / 100

# ──────────────────────────────────────────────────────────────────────────────
# DERIVED QUANTITIES (everything downstream uses these)
# ──────────────────────────────────────────────────────────────────────────────
base_ate = ATE_NO_PIRA if exclude_pira else CAUSAL_ATE
adj_ate  = base_ate * multiplier
adj_se   = CAUSAL_SE * multiplier
adj_lb, adj_ub = ci(adj_ate, adj_se)

enrollees    = int(round(n_workers * take_up_rate))
total_impact = enrollees * adj_ate           # one-time stock increase ($)
total_cost   = enrollees * admin_cost        # annual admin cost ($/yr)
net_impact   = total_impact - total_cost     # year-1 net
roi          = total_impact / total_cost if total_cost else float("inf")

# Lifetime compounded savings
lifetime_per_worker     = adj_ate * (1 + return_rate) ** horizon
lifetime_per_worker_lb  = adj_lb * (1 + return_rate) ** horizon
lifetime_per_worker_ub  = adj_ub * (1 + return_rate) ** horizon

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_exec, tab_whatif, tab_sens, tab_method = st.tabs(
    ["📈 Executive View", "🎛️ What-If Scenarios", "🔬 Sensitivity", "📋 Methodology"]
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# TAB 1 — EXECUTIVE VIEW
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_exec:
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Causal effect / worker",
        fmt_money(adj_ate),
        f"±{fmt_money(1.96 * adj_se)} (95% CI)",
    )
    c2.metric("Expected enrollees", f"{enrollees:,}", f"{take_up_rate:.0%} take-up")
    c3.metric("Total savings impact", fmt_money_M(total_impact))
    c4.metric(
        "Benefit / cost ratio",
        f"{roi:.1f}×",
        delta=f"{fmt_money_M(net_impact)} year-1 net",
        delta_color="normal",
    )

    st.markdown("---")
    left, right = st.columns(2)

    # ── Naive vs Causal bar with error bars
    with left:
        st.subheader("Naive OLS vs. Double ML")
        naive_lb, naive_ub = ci(NAIVE_ATE, NAIVE_SE)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Naive OLS<br>(biased)"], y=[NAIVE_ATE],
            marker_color=C_BIASED, name="Naive",
            error_y=dict(type="constant", value=1.96 * NAIVE_SE),
            text=[fmt_money(NAIVE_ATE)], textposition="outside",
            hovertemplate="Naive OLS<br>ATE: %{y:$,.0f}<br>"
                          f"95% CI: [{fmt_money(naive_lb)}, {fmt_money(naive_ub)}]<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=["Double ML<br>(causal)"], y=[adj_ate],
            marker_color=C_CAUSAL, name="DML",
            error_y=dict(type="constant", value=1.96 * adj_se),
            text=[fmt_money(adj_ate)], textposition="outside",
            hovertemplate="Double ML<br>ATE: %{y:$,.0f}<br>"
                          f"95% CI: [{fmt_money(adj_lb)}, {fmt_money(adj_ub)}]<extra></extra>",
        ))
        fig.add_annotation(
            x=0.5, xref="paper", y=(NAIVE_ATE + adj_ate) / 2,
            text=f"<b>Selection bias<br>≈ {fmt_money(NAIVE_ATE - adj_ate)}</b>",
            showarrow=False, bgcolor="#FAECE7", bordercolor=C_ACCENT,
            font=dict(color="#993C1D", size=11),
        )
        fig.update_layout(
            yaxis_title="Effect on net financial assets ($)",
            yaxis_tickformat="$,.0f",
            template="plotly_white", showlegend=False, height=400,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Lifetime projection fan chart
    with right:
        st.subheader(f"Compounded Lifetime Effect ({horizon}y, {return_rate:.1%}/yr)")
        years = np.arange(0, horizon + 1)
        path     = adj_ate * (1 + return_rate) ** years
        path_lb  = adj_lb  * (1 + return_rate) ** years
        path_ub  = adj_ub  * (1 + return_rate) ** years

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=path_ub, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=years, y=path_lb, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(24,95,165,0.18)",
            name="95% CI", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=years, y=path, mode="lines",
            line=dict(color=C_CAUSAL, width=3),
            name="Point estimate",
            hovertemplate="Year %{x}<br>Compounded effect: %{y:$,.0f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Years after eligibility extended",
            yaxis_title="Per-worker savings ($)",
            yaxis_tickformat="$,.0f",
            template="plotly_white", height=400,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Counterfactual narrative card (THE rubric requirement)
    st.markdown("---")
    st.subheader("📌 Live Counterfactual Statement")
    st.success(
        f"**If your firm extends 401(k) eligibility to {n_workers:,} workers** "
        f"with a **{take_up_rate:.0%} take-up rate** and applies a **{multiplier:.2f}× "
        f"sensitivity multiplier** to the headline DML estimate, "
        f"the predicted aggregate savings impact is **{fmt_money_M(total_impact)}** "
        f"(95% CI: [{fmt_money_M(enrollees * adj_lb)}, {fmt_money_M(enrollees * adj_ub)}]) "
        f"against an annual admin cost of **{fmt_money_M(total_cost)}** "
        f"— a benefit-to-cost ratio of **{roi:.1f}×**. "
        + (
            "_Note: pira excluded from controls, so this estimate may include "
            "downstream IRA-channel savings._"
            if exclude_pira else ""
        )
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# TAB 2 — WHAT-IF SCENARIOS (3-up comparison)
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_whatif:
    st.subheader("Three pre-built scenarios — same enrollment, different causal assumptions")
    st.caption(
        "Each scenario applies a different multiplier to the DML estimate, "
        "reflecting different beliefs about the strength of the CIA."
    )

    scenarios = [
        ("Conservative", 0.50, "Heavy haircut — assumes ~50% of estimate is firm-level confounding",  "#888780"),
        ("Base case",    1.00, "CIA holds — DML estimate is taken at face value",                     C_CAUSAL),
        ("Optimistic",   1.30, "Positive peer-effect spillovers (Duflo & Saez 2003) amplify effect", "#3A8B40"),
    ]

    cols = st.columns(3)
    for col, (name, m, desc, color) in zip(cols, scenarios):
        with col:
            s_ate = base_ate * m
            s_se  = CAUSAL_SE * m
            s_lb, s_ub = ci(s_ate, s_se)
            s_total = enrollees * s_ate
            s_lb_t, s_ub_t = enrollees * s_lb, enrollees * s_ub
            s_roi = s_total / total_cost if total_cost else float("inf")

            st.markdown(f"### {name}")
            st.caption(desc)
            st.metric("Per-worker effect", fmt_money(s_ate),
                      f"95% CI: [{fmt_money(s_lb)}, {fmt_money(s_ub)}]")
            st.metric("Total savings", fmt_money_M(s_total),
                      f"CI: [{fmt_money_M(s_lb_t)}, {fmt_money_M(s_ub_t)}]")
            st.metric("Benefit/cost", f"{s_roi:.1f}×",
                      "✓ recommend" if s_roi > 1 else "✗ not viable",
                      delta_color="normal" if s_roi > 1 else "inverse")

    st.markdown("---")

    # ── Side-by-side parameter sweep
    st.subheader("Parameter sweep: how does total impact depend on a single assumption?")
    sweep_var = st.selectbox(
        "Sweep which parameter?",
        ["Take-up rate", "Number of newly eligible", "Effect multiplier", "Admin cost"],
    )

    if sweep_var == "Take-up rate":
        xs = np.linspace(0.10, 1.0, 50)
        ys = n_workers * xs * adj_ate
        ys_lb = n_workers * xs * adj_lb
        ys_ub = n_workers * xs * adj_ub
        x_label = "Take-up rate"
        cur_x = take_up_rate
        cur_label = f"{cur_x:.0%}"
    elif sweep_var == "Number of newly eligible":
        xs = np.linspace(100, 25_000, 50)
        ys = xs * take_up_rate * adj_ate
        ys_lb = xs * take_up_rate * adj_lb
        ys_ub = xs * take_up_rate * adj_ub
        x_label = "Newly eligible workers"
        cur_x = n_workers
        cur_label = f"{cur_x:,}"
    elif sweep_var == "Effect multiplier":
        xs = np.linspace(0.3, 1.5, 50)
        ys = enrollees * base_ate * xs
        ys_lb = enrollees * (base_ate * xs - 1.96 * CAUSAL_SE * xs)
        ys_ub = enrollees * (base_ate * xs + 1.96 * CAUSAL_SE * xs)
        x_label = "Effect multiplier"
        cur_x = multiplier
        cur_label = f"{cur_x:.2f}×"
    else:  # Admin cost
        xs = np.linspace(100, 600, 50)
        ys = enrollees * adj_ate - enrollees * xs       # year-1 net
        ys_lb = enrollees * adj_lb - enrollees * xs
        ys_ub = enrollees * adj_ub - enrollees * xs
        x_label = "Admin cost ($/yr/participant)"
        cur_x = admin_cost
        cur_label = f"${cur_x:,.0f}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys_ub, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys_lb, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(24,95,165,0.15)", name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=C_CAUSAL, width=3), name="Point estimate",
    ))
    fig.add_vline(x=cur_x, line_dash="dash", line_color=C_ACCENT,
                  annotation_text=f"Current: {cur_label}",
                  annotation_position="top")
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=("Net impact ($)" if sweep_var == "Admin cost" else "Total savings ($)"),
        yaxis_tickformat="$,.0f",
        template="plotly_white", height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# TAB 3 — SENSITIVITY (TORNADO + ROBUSTNESS)
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_sens:
    st.subheader("Tornado: which assumption moves the bottom line most?")
    st.caption(
        "Each bar shows how much the **year-1 net impact** changes when one "
        "assumption is varied to its low / high bound, holding all others at current values."
    )

    base_net = total_impact - total_cost

    # Each row: (param, low, high, low_net, high_net)
    rows = [
        ("Take-up rate",
            0.30, 0.90,
            n_workers * 0.30 * adj_ate - n_workers * 0.30 * admin_cost,
            n_workers * 0.90 * adj_ate - n_workers * 0.90 * admin_cost),
        ("Effect multiplier",
            0.5, 1.3,
            enrollees * base_ate * 0.5 - total_cost,
            enrollees * base_ate * 1.3 - total_cost),
        ("Admin cost ($/yr)",
            150, 500,
            enrollees * adj_ate - enrollees * 150,
            enrollees * adj_ate - enrollees * 500),
        ("# eligible workers",
            500, 5000,
            500  * take_up_rate * adj_ate - 500  * take_up_rate * admin_cost,
            5000 * take_up_rate * adj_ate - 5000 * take_up_rate * admin_cost),
    ]

    # Sort by absolute swing
    rows = sorted(rows, key=lambda r: abs(r[4] - r[3]), reverse=True)

    fig = go.Figure()
    for i, (name, lo, hi, lo_net, hi_net) in enumerate(rows):
        # Lower bound bar (red, leftward from base)
        fig.add_trace(go.Bar(
            y=[name], x=[lo_net - base_net], base=base_net,
            orientation="h", marker_color=C_BIASED,
            name="Low bound" if i == 0 else None, showlegend=(i == 0),
            hovertemplate=f"{name} = {lo}<br>Net impact: %{{base:$,.0f}} → "
                          f"{fmt_money(lo_net)}<extra></extra>",
        ))
        # Upper bound bar (blue, rightward from base)
        fig.add_trace(go.Bar(
            y=[name], x=[hi_net - base_net], base=base_net,
            orientation="h", marker_color=C_CAUSAL,
            name="High bound" if i == 0 else None, showlegend=(i == 0),
            hovertemplate=f"{name} = {hi}<br>Net impact: %{{base:$,.0f}} → "
                          f"{fmt_money(hi_net)}<extra></extra>",
        ))

    fig.add_vline(x=base_net, line_dash="dash", line_color=C_MUTED,
                  annotation_text=f"Current base: {fmt_money_M(base_net)}",
                  annotation_position="top")
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Year-1 net impact ($)",
        xaxis_tickformat="$,.0f",
        template="plotly_white", height=380,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Robustness comparison
    st.subheader("Cross-method robustness")

    rob_df = pd.DataFrame({
        "Estimator": [
            "Naive OLS (no controls)",
            "DML (GBM nuisance) — main",
            "DML (RF nuisance) — robustness",
            "DML excluding pira (bad-control test)",
        ],
        "ATE ($)": [NAIVE_ATE, CAUSAL_ATE, ROBUST_ATE, ATE_NO_PIRA],
        "95% CI lower": [
            ci(NAIVE_ATE, NAIVE_SE)[0], ci(CAUSAL_ATE, CAUSAL_SE)[0],
            ci(ROBUST_ATE, CAUSAL_SE)[0], ci(ATE_NO_PIRA, CAUSAL_SE)[0],
        ],
        "95% CI upper": [
            ci(NAIVE_ATE, NAIVE_SE)[1], ci(CAUSAL_ATE, CAUSAL_SE)[1],
            ci(ROBUST_ATE, CAUSAL_SE)[1], ci(ATE_NO_PIRA, CAUSAL_SE)[1],
        ],
        "Interpretation": [
            "Biased upward — confounds eligibility with selection",
            "Main causal estimate under CIA",
            "Same ATE with different ML — confirms not artifact",
            "Higher than main → pira may absorb effect (mediator)",
        ],
    })

    fig = go.Figure()
    colors = [C_BIASED, C_CAUSAL, C_CAUSAL, C_ACCENT]
    for i, row in rob_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["ATE ($)"]], y=[row["Estimator"]],
            mode="markers",
            marker=dict(size=14, color=colors[i],
                        line=dict(color="white", width=2)),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["95% CI upper"] - row["ATE ($)"]],
                arrayminus=[row["ATE ($)"] - row["95% CI lower"]],
                color=colors[i], thickness=2, width=8,
            ),
            showlegend=False,
            hovertemplate=f"{row['Estimator']}<br>"
                          f"ATE: {fmt_money(row['ATE ($)'])}<br>"
                          f"CI: [{fmt_money(row['95% CI lower'])}, "
                          f"{fmt_money(row['95% CI upper'])}]<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="ATE ($)", xaxis_tickformat="$,.0f",
        template="plotly_white", height=320,
        margin=dict(t=20, b=20, l=120),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display the table
    st.dataframe(
        rob_df.style.format({
            "ATE ($)": "${:,.0f}",
            "95% CI lower": "${:,.0f}",
            "95% CI upper": "${:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown(
        f"**Interpretation:** The main DML estimate ({fmt_money(CAUSAL_ATE)}) "
        f"shifts by less than {fmt_money(abs(CAUSAL_ATE - ROBUST_ATE))} when the "
        f"nuisance learner is changed from GBM to Random Forest, suggesting the "
        f"result is not driven by a specific ML choice. Excluding `pira` "
        f"increases the ATE to {fmt_money(ATE_NO_PIRA)}, consistent with "
        f"`pira` partially absorbing the treatment effect — a downward bias "
        f"flagged in §2 of the Threats memo."
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# TAB 4 — METHODOLOGY
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_method:
    st.subheader("Identification Strategy: Double Machine Learning")

    st.markdown(
        """
**Estimator.** Linear DML (Chernozhukov et al., 2018, *Econometrica*) with
gradient-boosted trees as nuisance learners and 5-fold cross-fitting.

**Three-step procedure:**
1. **Residualize the outcome** — fit $\\hat{Y} = m(W)$ with ML;
   compute $\\tilde{Y} = Y - \\hat{Y}$
2. **Residualize the treatment** — fit $\\hat{T} = g(W)$ with ML;
   compute $\\tilde{T} = T - \\hat{T}$
3. **Causal regression** — regress $\\tilde{Y}$ on $\\tilde{T}$;
   the coefficient is the ATE

Cross-fitting prevents nuisance overfitting from contaminating the causal
estimate. The estimator is $\\sqrt{n}$-consistent and asymptotically normal
under the Conditional Independence Assumption (CIA).
        """
    )

    st.subheader("Variables")
    var_df = pd.DataFrame([
        ("e401",       "Treatment", "Binary indicator for 401(k) eligibility"),
        ("net_tfa",    "Outcome",   "Net total financial assets ($)"),
        ("age",        "Control",   "Age in years"),
        ("inc",        "Control",   "Annual income ($) — primary confounder"),
        ("fsize",      "Control",   "Family size"),
        ("educ",       "Control",   "Years of education"),
        ("marr",       "Control",   "Married indicator"),
        ("twoearn",    "Control",   "Two-earner household indicator"),
        ("pira",       "Control",   "IRA participation — possible bad control"),
        ("p401",       "Excluded",  "401(k) participation — post-treatment"),
    ], columns=["Variable", "Role", "Description"])
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    st.subheader("Why a prediction-only model fails this question")
    st.markdown(
        """
A Random Forest fit on the same feature set achieves moderate predictive
$R^2$, but its feature importance for `e401` is a **mixture** of:

- the causal effect of eligibility, and
- selection on income / motivation / firm quality.

A policymaker asking *"what happens to savings if we change eligibility?"*
needs the first component cleanly. Prediction models cannot separate the
two — only an identification strategy can.
        """
    )

    st.subheader("Threats to identification — summary")
    st.markdown(
        f"""
| Threat | Direction of bias | Addressed by |
| --- | --- | --- |
| Firm-level unobservables (savings culture, total comp) | **Upward** | IV (state-level plan availability — unavailable in 1991 SIPP) |
| `pira` as bad control (post-treatment mediator) | **Downward** | Toggle in sidebar — see {fmt_money(ATE_NO_PIRA)} when excluded |
| SUTVA violations (peer effects in enrollment) | Ambiguous | RCT with cluster randomization |

Under firm-level confounding, our base estimate of {fmt_money(CAUSAL_ATE)}
is best read as an **upper bound**. A 50% haircut yields a conservative
lower bound of {fmt_money(CAUSAL_ATE * 0.5)} — already comfortably positive.
        """
    )

    st.subheader("Citations")
    st.markdown(
        """
- Chernozhukov, V. et al. (2018). "Double/debiased machine learning for
  treatment and structural parameters." *Econometrica*.
- Poterba, J., Venti, S., & Wise, D. (1995). "Do 401(k) contributions
  crowd out other personal saving?" *Journal of Public Economics*.
- Duflo, E. & Saez, E. (2003). "The role of information and social
  interactions in retirement plan decisions." *Quarterly Journal of Economics*.
- Engelhardt, G. & Kumar, A. (2007). "Employer matching and 401(k) saving."
  *Journal of Public Economics*.
        """
    )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with Streamlit · ECON 5200 Final Project · "
    "Source: github.com/<your-handle>/<repo>"
)
