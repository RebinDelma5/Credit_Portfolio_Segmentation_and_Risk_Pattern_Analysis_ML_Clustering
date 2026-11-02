"""Credit Portfolio Segmentation - simple Streamlit dashboard.
Run it from the project folder:
    streamlit run app.py

This app does NOT train anything. It only reads files that the notebook created:
    models/model.pkl                          the trained model (saved with pickle)
    data/processed/dashboard_sample.csv       10,000 sample borrowers + map coordinates
    data/processed/app_cluster_summary.csv    profile + default rate of each segment
    data/processed/app_default_by_year.csv    default rate by segment and year
    data/processed/app_algorithm_comparison.csv
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------
# 1. Page settings
# ---------------------------------------------------------------
st.set_page_config(page_title="Credit Portfolio Segmentation", layout="wide")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# 2. Load the files created by the notebook
# ---------------------------------------------------------------
summary = pd.read_csv("data/processed/app_cluster_summary.csv")
sample = pd.read_csv("data/processed/dashboard_sample.csv")
by_year = pd.read_csv("data/processed/app_default_by_year.csv", index_col=0)
algos = pd.read_csv("data/processed/app_algorithm_comparison.csv")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

names = list(summary["segment"])               # the 5 segment names
colors = sns.color_palette("tab10", 5)         # one colour for each segment

# Default rate of ALL borrowers = total defaults / total borrowers
overall_rate = summary["defaults"].sum() / summary["borrowers"].sum() * 100

# ---------------------------------------------------------------
# 3. Title and tabs
# ---------------------------------------------------------------
st.title("Credit Portfolio Segmentation")
st.caption("Borrower segments found with K-Means. The Default column was NOT used to "
           "build the segments. It is only used to see how each group repaid.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Segments", "Default rates", "Map", "Try a borrower", "Insights"])

# ---------------------------------------------------------------
# 4. Overview tab
# ---------------------------------------------------------------
with tab1:
    kmeans_row = algos[algos["algorithm"] == "K-Means"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Borrowers", f"{int(summary['borrowers'].sum()):,}")
    col2.metric("Default rate", f"{overall_rate:.1f}%")
    col3.metric("Segments", len(summary))
    col4.metric("Silhouette score", f"{kmeans_row['silhouette']:.3f}")

    left, right = st.columns(2)

    # Pie chart: how big is each segment?
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.pie(summary["borrowers"], labels=names, colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title("Share of borrowers in each segment")
    left.pyplot(fig)
    plt.close(fig)

    # Bar chart: default rate of each segment
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, summary["default_rate_pct"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%")
    ax.axhline(overall_rate, color="black", linestyle="--", label=f"average {overall_rate:.1f}%")
    ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate by segment")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    right.pyplot(fig)
    plt.close(fig)

    st.subheader("Which algorithm worked best?")
    st.dataframe(algos, hide_index=True)
    st.caption("Silhouette: higher is better. K-Means gave the clearest segments. "
               "DBSCAN found one dense group plus a little noise, so it is used to find anomalies.")

# ---------------------------------------------------------------
# 5. Segments tab
# ---------------------------------------------------------------
with tab2:
    chosen = st.selectbox("Choose a segment", names)
    row = summary[summary["segment"] == chosen].iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Borrowers", f"{int(row['borrowers']):,}")
    col2.metric("Median income", f"${row['median_income']:,.0f}")
    col3.metric("Median FICO", f"{row['median_fico']:.0f}")
    col4.metric("Median loan", f"${row['median_loan_amnt']:,.0f}")
    col5.metric("Default rate", f"{row['default_rate_pct']:.1f}%")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Median DTI", f"{row['median_dti']:.1f}")
    col2.metric("Average job years", f"{row['mean_emp_years']:.1f}")
    col3.metric("Loan / income", f"{row['median_loan_to_income']:.2f}")
    col4.metric("Renters", f"{row['pct_rent']:.0f}%")
    col5.metric("Debt consolidation", f"{row['pct_debt_consolidation']:.0f}%")

    st.subheader("Compare all segments")
    measure = st.selectbox("Pick a measure", ["median_income", "median_fico", "median_dti",
                                              "median_loan_amnt", "mean_emp_years",
                                              "median_loan_to_income", "pct_rent",
                                              "default_rate_pct"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, summary[measure], color=colors)
    ax.set_ylabel(measure)
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Spread inside each segment (10,000-borrower sample)")
    feature = st.selectbox("Feature", ["fico_n", "revenue", "dti_n", "loan_amnt", "emp_years"])
    data = sample
    if feature == "revenue":
        # hide a few huge incomes so the boxes are readable
        data = sample[sample["revenue"] < sample["revenue"].quantile(0.99)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=data, x="cluster_name", y=feature, order=names,
                hue="cluster_name", hue_order=names, palette=colors, legend=False, ax=ax)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------
# 6. Default rates tab
# ---------------------------------------------------------------
with tab3:
    st.write("Default was never used to make the segments, yet their default rates are "
             "very different. So the segments are meaningful.")

    # Error bars show the 95% confidence interval
    below = summary["default_rate_pct"] - summary["ci_low_pct"]
    above = summary["ci_high_pct"] - summary["default_rate_pct"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, summary["default_rate_pct"], yerr=[below, above], capsize=5, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=8)
    ax.axhline(overall_rate, color="black", linestyle="--", label="average of all borrowers")
    ax.set_ylabel("Default rate (%)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Is the order the same every year?")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(by_year.columns):
        ax.plot(by_year.index, by_year[name], marker="o", color=colors[i], label=name)
    ax.set_xlabel("Issue year")
    ax.set_ylabel("Default rate (%)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Prime Credit is the lowest in every year 2008-2018. Only years with at least "
               "200 loans per segment are shown. Loans from 2017-2018 look safer only because "
               "many have not had time to default yet.")

# ---------------------------------------------------------------
# 7. Map tab (PCA, t-SNE and UMAP squeeze many features into 2-D)
# ---------------------------------------------------------------
with tab4:
    st.write("Each dot is one borrower. Dots that are close together are similar borrowers.")
    col1, col2 = st.columns(2)
    method = col1.radio("Method", ["PCA", "t-SNE", "UMAP"], horizontal=True)
    colour_by = col2.radio("Colour by", ["Segment", "Repaid / Default"], horizontal=True)

    if method == "PCA":
        x_col, y_col = "pca_1", "pca_2"
    elif method == "t-SNE":
        x_col, y_col = "tsne_1", "tsne_2"
    else:
        x_col, y_col = "umap_1", "umap_2"

    fig, ax = plt.subplots(figsize=(9, 6.5))
    if colour_by == "Segment":
        sns.scatterplot(data=sample, x=x_col, y=y_col, hue="cluster_name", hue_order=names,
                        palette=colors, s=8, alpha=0.6, linewidth=0, ax=ax)
    else:
        sample["Outcome"] = sample["Default"].map({0: "Repaid", 1: "Default"})
        sns.scatterplot(data=sample, x=x_col, y=y_col, hue="Outcome",
                        palette={"Repaid": "#9ecae1", "Default": "#d62728"},
                        s=8, alpha=0.6, linewidth=0, ax=ax)
    ax.set_xlabel(method + " 1")
    ax.set_ylabel(method + " 2")
    ax.legend(markerscale=2, fontsize=8)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The segments overlap at the edges because real borrowers change gradually.")

# ---------------------------------------------------------------
# 8. Try a borrower tab (uses the pickled model)
# ---------------------------------------------------------------
with tab5:
    st.warning("Learning demo only. It shows which segment a profile is closest to and that "
               "segment's HISTORICAL default rate. It is not a credit decision.")

    col1, col2, col3 = st.columns(3)
    income = col1.number_input("Annual income ($)", 5000, 1000000, 65000, 1000)
    dti = col2.number_input("Debt-to-income ratio", 0.0, 100.0, 17.0, 0.5)
    loan = col3.number_input("Loan amount ($)", 500, 40000, 12000, 500)

    col1, col2, col3 = st.columns(3)
    fico = col1.slider("FICO score", 600, 850, 690)
    emp_length = col2.selectbox("Employment length", list(model["emp_length_map"]), index=5)
    home = col3.selectbox("Home ownership", ["MORTGAGE", "RENT", "OWN"])
    purpose = st.selectbox("Loan purpose", ["debt_consolidation", "credit_card", "home_improvement",
                                            "other", "major_purchase", "medical", "small_business",
                                            "car", "moving", "vacation", "house", "wedding",
                                            "renewable_energy", "educational"])

    # Step 1: put the inputs in a one-row table
    person = pd.DataFrame({"revenue": [income], "dti_n": [dti], "loan_amnt": [loan],
                           "fico_n": [fico], "purpose": [purpose], "home_ownership_n": [home],
                           "emp_years": [model["emp_length_map"][emp_length]]})

    # Step 2: build the same features that were used when the model was trained
    caps = model["caps"]
    person["log_income"] = np.log1p(person["revenue"].clip(upper=caps["revenue"]))
    person["dti_n"] = person["dti_n"].clip(upper=caps["dti_n"])
    person["loan_to_income"] = (person["loan_amnt"] / person["revenue"]).clip(upper=caps["loan_to_income"])
    if purpose in ["debt_consolidation", "credit_card", "home_improvement"]:
        person["purpose_group"] = purpose
    else:
        person["purpose_group"] = "other"

    # Step 3: scale -> PCA -> K-Means -> segment name
    scaled = model["preprocessor"].transform(person[model["feature_columns"]])
    reduced = model["pca"].transform(scaled)
    kmeans_number = model["kmeans"].predict(reduced)[0]
    segment_number = model["label_map"][kmeans_number]
    segment = model["cluster_names"][segment_number]

    seg_row = summary[summary["segment"] == segment].iloc[0]
    st.subheader("Closest segment: " + segment)
    st.write(f"In the past, **{seg_row['default_rate_pct']:.1f}%** of borrowers in this "
             f"segment defaulted (all borrowers: {overall_rate:.1f}%).")

# ---------------------------------------------------------------
# 9. Insights tab
# ---------------------------------------------------------------
with tab6:
    low = summary.loc[summary["default_rate_pct"].idxmin()]
    high = summary.loc[summary["default_rate_pct"].idxmax()]
    ratio = high["default_rate_pct"] / low["default_rate_pct"]

    st.subheader("What we learned")
    st.write(f"* Default rates differ a lot: from **{low['default_rate_pct']:.1f}%** "
             f"({low['segment']}) to **{high['default_rate_pct']:.1f}%** ({high['segment']}), "
             f"about {ratio:.1f} times higher.")
    st.write("* Credit score matters, but so does affordability. *Stretched Low-Income* borrowers "
             "have an ordinary FICO score but the highest debt-to-income and loan-to-income.")
    st.write("* Anomalies (unusual borrowers found by DBSCAN) are mostly very high credit scores "
             "or incomes. They are unusual, not necessarily risky.")

    st.subheader("How a bank could use this")
    st.write("* Pricing and limits: tighter loan-to-income limits for *Stretched Low-Income* applicants.")
    st.write("* Products: small starter loans for *Early-Career Renters*; fast, low-rate offers for *Prime Credit*.")
    st.write("* Monitoring: watch the share of each segment every quarter.")

    st.subheader("Limitations")
    st.write("* Silhouette scores are low (about 0.15): segments overlap. They are tendencies, not hard borders.")
    st.write("* Recent loans (2017-2018) have not had time to default.")
    st.write("* Segments describe groups. Never judge one person only by their segment.")
