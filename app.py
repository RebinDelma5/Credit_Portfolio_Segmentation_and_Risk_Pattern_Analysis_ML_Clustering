"""Credit Portfolio Segmentation - simple Streamlit dashboard.

How to run (from the project folder):
    streamlit run app.py

The app only READS files that the notebook created:
    models/model.pkl                       -> the trained model (saved with pickle)
    data/processed/dashboard_sample.csv    -> 10,000 sample borrowers + 2-D map coordinates
    data/processed/app_cluster_summary.csv -> profile and default rate of each segment
    data/processed/app_default_by_year.csv -> default rate by segment and year
    data/processed/app_algorithm_comparison.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# 1. Page settings and file paths
# ----------------------------------------------------------------------
st.set_page_config(page_title="Credit Portfolio Segmentation", page_icon="📊", layout="wide")

BASE = Path(__file__).parent                 # folder that contains app.py
DATA = BASE / "data" / "processed"


# ----------------------------------------------------------------------
# 2. Loading data (st.cache_* means: load once, then reuse = faster app)
# ----------------------------------------------------------------------
@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(DATA / name, **kwargs)


@st.cache_resource
def load_model():
    """Load the trained model that the notebook saved with pickle."""
    with open(BASE / "models" / "model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
summary = load_csv("app_cluster_summary.csv")          # one row per segment
sample = load_csv("dashboard_sample.csv")              # 10,000 sample borrowers
by_year = load_csv("app_default_by_year.csv", index_col=0)
algos = load_csv("app_algorithm_comparison.csv")

names = list(summary["segment"])                       # segment names, ordered by FICO
colors = dict(zip(names, px.colors.qualitative.D3))    # one fixed colour per segment
sample["cluster_name"] = pd.Categorical(sample["cluster_name"], names, ordered=True)

# Portfolio default rate = total defaults / total borrowers
overall_rate = summary["defaults"].sum() / summary["borrowers"].sum() * 100

# ----------------------------------------------------------------------
# 3. Title and tabs
# ----------------------------------------------------------------------
st.title("📊 Credit Portfolio Segmentation")
st.caption("Borrower segments found with K-Means. The Default column was NOT used to "
           "build the segments - it is only used to compare how each group repaid.")

tab_overview, tab_segments, tab_default, tab_map, tab_try, tab_insights = st.tabs(
    ["Overview", "Segments", "Default rates", "Map", "Try a borrower", "Insights"])

# ----------------------------------------------------------------------
# 4. Overview tab
# ----------------------------------------------------------------------
with tab_overview:
    kmeans_row = algos[algos["algorithm"] == "K-Means"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers", f"{int(summary['borrowers'].sum()):,}")
    c2.metric("Default rate", f"{overall_rate:.1f}%")
    c3.metric("Segments", len(summary))
    c4.metric("Silhouette score", f"{kmeans_row['silhouette']:.3f}")

    left, right = st.columns(2)
    with left:
        fig = px.pie(summary, values="borrowers", names="segment", hole=0.45,
                     color="segment", color_discrete_map=colors,
                     title="Share of borrowers in each segment")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(summary, x="segment", y="default_rate_pct", color="segment",
                     color_discrete_map=colors, text=summary["default_rate_pct"].round(1),
                     title="Default rate (%) by segment")
        fig.add_hline(y=overall_rate, line_dash="dash",
                      annotation_text=f"average {overall_rate:.1f}%")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Default rate (%)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Which algorithm worked best?")
    st.dataframe(algos[["algorithm", "n_clusters", "noise_pct", "silhouette",
                        "davies_bouldin", "ARI_vs_kmeans"]], width="stretch", hide_index=True)
    st.caption("Silhouette: higher is better. K-Means gave the clearest segments. "
               "DBSCAN found one dense group plus a little noise, so it is used for anomalies.")

# ----------------------------------------------------------------------
# 5. Segments tab
# ----------------------------------------------------------------------
with tab_segments:
    chosen = st.selectbox("Choose a segment", names)
    row = summary[summary["segment"] == chosen].iloc[0]

    c = st.columns(5)
    c[0].metric("Borrowers", f"{int(row['borrowers']):,}", f"{row['share_pct']:.1f}% of all")
    c[1].metric("Median income", f"${row['median_income']:,.0f}")
    c[2].metric("Median FICO", f"{row['median_fico']:.0f}")
    c[3].metric("Median loan", f"${row['median_loan_amnt']:,.0f}")
    c[4].metric("Default rate", f"{row['default_rate_pct']:.1f}%")

    c = st.columns(5)
    c[0].metric("Median DTI", f"{row['median_dti']:.1f}")
    c[1].metric("Avg. job years", f"{row['mean_emp_years']:.1f}")
    c[2].metric("Loan / income", f"{row['median_loan_to_income']:.2f}")
    c[3].metric("Renters", f"{row['pct_rent']:.0f}%")
    c[4].metric("Debt consolidation", f"{row['pct_debt_consolidation']:.0f}%")

    st.subheader("Compare all segments")
    metric = st.selectbox("Pick a measure", [col for col in summary.columns
                                             if col not in ("segment", "borrowers", "defaults")],
                          index=3)
    fig = px.bar(summary, x="segment", y=metric, color="segment", color_discrete_map=colors)
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Spread inside each segment (10,000-borrower sample)")
    feature = st.selectbox("Feature", ["fico_n", "revenue", "dti_n", "loan_amnt", "emp_years"])
    data = sample
    if feature == "revenue":                          # hide a few huge incomes so the plot is readable
        data = sample[sample["revenue"] <= sample["revenue"].quantile(0.99)]
    fig = px.box(data, x="cluster_name", y=feature, color="cluster_name",
                 color_discrete_map=colors, category_orders={"cluster_name": names})
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------
# 6. Default rates tab
# ----------------------------------------------------------------------
with tab_default:
    st.write("Default was never used to make the segments, yet their default rates are "
             "very different - so the segments are meaningful.")

    # error bars show the 95% confidence interval
    fig = px.bar(summary, x="segment", y="default_rate_pct", color="segment",
                 color_discrete_map=colors, text=summary["default_rate_pct"].round(1),
                 error_y=summary["ci_high_pct"] - summary["default_rate_pct"],
                 error_y_minus=summary["default_rate_pct"] - summary["ci_low_pct"])
    fig.add_hline(y=overall_rate, line_dash="dash", annotation_text="average")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Default rate (%)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Is the order the same every year?")
    fig = px.line(by_year, markers=True, color_discrete_map=colors,
                  labels={"value": "Default rate (%)", "issue_year": "Issue year",
                          "variable": "Segment"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Prime Credit is the lowest in every year 2008-2018. Only years with at least "
               "200 loans per segment are shown. 2017-2018 loans are less 'mature', so their "
               "rates look lower.")

# ----------------------------------------------------------------------
# 7. Map tab (PCA / t-SNE / UMAP squeeze many features into 2-D)
# ----------------------------------------------------------------------
with tab_map:
    st.write("Each dot is one borrower. Dots that are close together are similar borrowers.")
    c1, c2 = st.columns(2)
    method = c1.radio("Method", ["PCA", "t-SNE", "UMAP"], horizontal=True)
    colour_by = c2.radio("Colour by", ["Segment", "Repaid / Default"], horizontal=True)

    x_col, y_col = {"PCA": ("pca_1", "pca_2"), "t-SNE": ("tsne_1", "tsne_2"),
                    "UMAP": ("umap_1", "umap_2")}[method]

    plot_data = sample.copy()
    plot_data["Outcome"] = plot_data["Default"].map({0: "Repaid", 1: "Default"})
    if colour_by == "Segment":
        fig = px.scatter(plot_data, x=x_col, y=y_col, color="cluster_name",
                         color_discrete_map=colors, category_orders={"cluster_name": names},
                         render_mode="webgl", opacity=0.6, height=600)
    else:
        fig = px.scatter(plot_data, x=x_col, y=y_col, color="Outcome",
                         color_discrete_map={"Repaid": "#9ecae1", "Default": "#d62728"},
                         render_mode="webgl", opacity=0.6, height=600)
    fig.update_traces(marker_size=5)
    st.plotly_chart(fig, width="stretch")
    st.caption("Segments overlap at the edges because real borrowers change gradually.")

# ----------------------------------------------------------------------
# 8. Try a borrower tab (uses the pickled model)
# ----------------------------------------------------------------------
with tab_try:
    st.warning("Learning demo only. It shows which segment a profile is closest to and that "
               "segment's HISTORICAL default rate. It is not a credit decision.")

    c1, c2, c3 = st.columns(3)
    income = c1.number_input("Annual income ($)", 5_000, 1_000_000, 65_000, 1_000)
    dti = c2.number_input("Debt-to-income ratio", 0.0, 100.0, 17.0, 0.5)
    loan = c3.number_input("Loan amount ($)", 500, 40_000, 12_000, 500)

    c1, c2, c3 = st.columns(3)
    fico = c1.slider("FICO score", 600, 850, 690)
    emp_length = c2.selectbox("Employment length", list(model["emp_length_map"]), index=5)
    home = c3.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN"])
    purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card",
                                            "home_improvement", "other", "major_purchase",
                                            "medical", "small_business", "car", "moving",
                                            "vacation", "house", "wedding",
                                            "renewable_energy", "educational"])

    # --- build the same features that were used when the model was trained
    caps = model["caps"]
    person = pd.DataFrame([{"revenue": income, "dti_n": dti, "loan_amnt": loan, "fico_n": fico,
                            "purpose": purpose, "home_ownership_n": home,
                            "emp_years": model["emp_length_map"][emp_length]}])
    person["log_income"] = np.log1p(person["revenue"].clip(upper=caps["revenue"]))
    person["dti_n"] = person["dti_n"].clip(upper=caps["dti_n"])
    person["loan_to_income"] = (person["loan_amnt"] / person["revenue"]).clip(upper=caps["loan_to_income"])
    person["purpose_group"] = np.where(
        person["purpose"].isin(["debt_consolidation", "credit_card", "home_improvement"]),
        person["purpose"], "other")

    # --- scale -> PCA -> K-Means -> segment name
    scaled = model["preprocessor"].transform(person[model["feature_columns"]])
    reduced = model["pca"].transform(scaled)
    segment_id = int(model["label_map"][model["kmeans"].predict(reduced)[0]])
    segment = model["cluster_names"][segment_id]

    seg_row = summary[summary["segment"] == segment].iloc[0]
    st.subheader(f"Closest segment: {segment}")
    st.write(f"In the past, **{seg_row['default_rate_pct']:.1f}%** of borrowers in this "
             f"segment defaulted (all borrowers: {overall_rate:.1f}%).")

# ----------------------------------------------------------------------
# 9. Insights tab
# ----------------------------------------------------------------------
with tab_insights:
    low = summary.loc[summary["default_rate_pct"].idxmin()]
    high = summary.loc[summary["default_rate_pct"].idxmax()]

    st.subheader("What we learned")
    st.markdown(f"""
* **Default rates differ a lot:** from **{low['default_rate_pct']:.1f}%** ({low['segment']}) to
  **{high['default_rate_pct']:.1f}%** ({high['segment']}), about
  {high['default_rate_pct'] / low['default_rate_pct']:.1f}x higher.
* **Credit score matters, but so does affordability.** *Stretched Low-Income* borrowers have an
  ordinary FICO score but the highest debt-to-income and loan-to-income.
* **Anomalies** (unusual borrowers found by DBSCAN) are mostly very high credit scores or incomes -
  unusual, not necessarily risky.
""")
    st.subheader("How the bank could use this")
    st.markdown("""
* **Pricing and limits:** tighter loan-to-income limits for *Stretched Low-Income* applicants.
* **Products:** small starter loans for *Early-Career Renters*; fast, low-rate offers for *Prime Credit*.
* **Monitoring:** watch the share of each segment every quarter.
""")
    st.subheader("Limitations")
    st.markdown("""
* Silhouette scores are low (about 0.15): segments overlap, they are tendencies, not hard borders.
* Recent loans (2017-2018) have not had time to default.
* Segments describe **groups** - never judge one person only by their segment.
""")
