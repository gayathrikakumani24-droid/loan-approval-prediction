import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import numpy as np

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="LoanIQ — Smart Credit Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================================
# DESIGN SYSTEM
# ====================================
PALETTE = {
    "navy_deep":   "#050D1A",
    "navy":        "#0A1628",
    "navy_mid":    "#0F2040",
    "navy_card":   "#122140",
    "steel":       "#1A2E50",
    "accent":      "#3B82F6",        # electric blue
    "accent_glow": "rgba(59,130,246,0.18)",
    "gold":        "#F59E0B",
    "success":     "#10B981",
    "danger":      "#EF4444",
    "text_primary":"#F1F5F9",
    "text_muted":  "#94A3B8",
    "border":      "#334155",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

* {{ box-sizing: border-box; }}

.stApp {{
    background: {PALETTE['navy_deep']};
    font-family: 'Inter', sans-serif;
}}

[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {PALETTE['navy']}; }}

/* Tabs */
[data-baseweb="tab-list"] {{
    background: {PALETTE['navy_mid']};
    border-radius: 14px;
    padding: 5px;
    border: 1px solid {PALETTE['border']};
    gap: 4px;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: {PALETTE['text_muted']} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    transition: all 0.2s !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: {PALETTE['accent']} !important;
    color: white !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.35) !important;
}}
[data-testid="stTabContent"] {{ padding-top: 24px; }}

/* Inputs */
label, .stNumberInput label, .stSelectbox label, .stSlider label {{
    color: {PALETTE['text_muted']} !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}}
.stNumberInput input {{
    background: {PALETTE['steel']} !important;
    color: {PALETTE['text_primary']} !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 15px !important;
}}
[data-baseweb="select"] > div {{
    background: {PALETTE['steel']} !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 10px !important;
    color: {PALETTE['text_primary']} !important;
}}
[data-baseweb="select"] span {{
    color: {PALETTE['text_primary']} !important;
}}

/* Slider */
[data-testid="stSlider"] .st-emotion-cache-1inwz65 {{
    color: {PALETTE['accent']} !important;
}}

/* Button */
.stButton > button {{
    background: linear-gradient(135deg, {PALETTE['accent']}, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(59,130,246,0.35) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 26px rgba(59,130,246,0.5) !important;
}}

/* Headings */
h1, h2, h3, h4 {{ color: {PALETTE['text_primary']} !important; font-family: 'Inter', sans-serif !important; }}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {PALETTE['navy_card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 16px 20px;
}}
[data-testid="metric-container"] label {{
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {PALETTE['text_primary']} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 26px !important;
}}

/* Divider */
hr {{ border-color: {PALETTE['border']} !important; }}

/* Matplotlib / pyplot */
.stPlotlyChart, .stpyplot {{ background: transparent !important; }}

/* DataFrame */
.stDataFrame {{ background: {PALETTE['navy_card']} !important; border-radius: 12px; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {PALETTE['navy']}; }}
::-webkit-scrollbar-thumb {{ background: {PALETTE['steel']}; border-radius: 4px; }}

/* Progress bar */
.stProgress > div > div {{
    background: linear-gradient(90deg, {PALETTE['accent']}, {PALETTE['gold']}) !important;
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)

# ====================================
# HELPERS
# ====================================

def card(content_html, bg=None, border=None, padding="24px", radius="16px", shadow=""):
    st.markdown(f"""
<div style="
    background:{PALETTE['navy_card']};
    border:1px solid {PALETTE['border']};
    border-radius:16px;
    padding:18px 20px;
    margin-bottom:16px;
">
    <div style="
        font-size:13px;
        font-weight:600;
        color:{PALETTE['text_muted']};
        letter-spacing:0.07em;
        text-transform:uppercase;
    ">
        Approval Split
    </div>
</div>
""", unsafe_allow_html=True)

def badge(text, color):
    return f"""<span style="
        background:{color}22;
        color:{color};
        border:1px solid {color}44;
        border-radius:20px;
        padding:3px 12px;
        font-size:12px;
        font-weight:600;
        letter-spacing:0.04em;
    ">{text}</span>"""

def fmt_inr(val):
    if val >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.1f} Cr"
    elif val >= 1_00_000:
        return f"₹{val/1_00_000:.1f} L"
    else:
        return f"₹{val:,.0f}"

# ====================================
# LOAD ASSETS
# ====================================
@st.cache_resource
def load_model():
    model = joblib.load("trained_model.pkl")
    columns = joblib.load("feature_columns.pkl")
    return model, columns

@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_dataset (1).csv")
    df.columns = df.columns.str.strip()
    df['education'] = df['education'].str.strip()
    df['self_employed'] = df['self_employed'].str.strip()
    df['loan_status'] = df['loan_status'].str.strip()
    df['total_assets'] = (
        df['residential_assets_value'] + df['commercial_assets_value']
        + df['luxury_assets_value'] + df['bank_asset_value']
    )
    return df

try:
    model, columns = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    df = None

# ====================================
# HEADER
# ====================================
st.markdown(f"""
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:28px 8px 8px 8px;
    border-bottom:1px solid {PALETTE['border']};
    margin-bottom:28px;
">
    <div>
        <div style="
            font-size:26px;
            font-weight:800;
            color:{PALETTE['text_primary']};
            letter-spacing:-0.5px;
            font-family:'Inter',sans-serif;
        ">
            🏦 LoanIQ
            <span style="
                font-size:13px;
                font-weight:500;
                color:{PALETTE['accent']};
                margin-left:10px;
                letter-spacing:0.06em;
            ">CREDIT ASSESSMENT PLATFORM</span>
        </div>
        <div style="color:{PALETTE['text_muted']};font-size:14px;margin-top:4px;">
            AI-powered eligibility analysis · Instant decisions · Transparent scoring
        </div>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
        {badge("v2.0", PALETTE['accent'])}
        {badge("LIVE", PALETTE['success'])}
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================
# TABS
# ====================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🎯  Loan Assessment",
    "📈  Dataset Analytics",
    "🔬  Feature Insights",
])

# ────────────────────────────────────
# TAB 1 — OVERVIEW / DASHBOARD
# ────────────────────────────────────
with tab1:
    if df is not None:
        total = len(df)
        approved = (df['loan_status'] == 'Approved').sum()
        rejected = total - approved
        avg_cibil = df['cibil_score'].mean()
        avg_income = df['income_annum'].mean()
        avg_loan = df['loan_amount'].mean()

        # KPI Row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Applications", f"{total:,}")
        k2.metric("Approved", f"{approved:,}", f"{approved/total*100:.1f}%")
        k3.metric("Rejected", f"{rejected:,}", f"-{rejected/total*100:.1f}%")
        k4.metric("Avg CIBIL Score", f"{avg_cibil:.0f}")
        k5.metric("Avg Loan Amount", fmt_inr(avg_loan))

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        c1, c2, c3 = st.columns(3)

        with c1:
            card(f"""<div style="font-size:13px;font-weight:600;color:{PALETTE['text_muted']};
                letter-spacing:0.07em;text-transform:uppercase;margin-bottom:14px;">
                Approval Split</div>""", padding="18px 20px 0px 20px")
            fig, ax = plt.subplots(figsize=(4, 4), facecolor="none")
            sizes = [approved, rejected]
            colors = [PALETTE['success'], PALETTE['danger']]
            wedges, texts, autotexts = ax.pie(
                sizes, autopct='%1.1f%%', startangle=90,
                colors=colors, pctdistance=0.75,
                wedgeprops=dict(width=0.55, edgecolor=PALETTE['navy_deep'], linewidth=2)
            )
            for t in autotexts:
                t.set_color('white'); t.set_fontsize(12); t.set_fontweight('bold')
            ax.set_facecolor("none")
            centre_circle = plt.Circle((0, 0), 0.45, fc=PALETTE['navy_card'])
            ax.add_patch(centre_circle)
            ax.text(0, 0, f"{approved/total*100:.0f}%\nApproved", ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white', family='monospace')
            legend_patches = [
                mpatches.Patch(color=PALETTE['success'], label='Approved'),
                mpatches.Patch(color=PALETTE['danger'], label='Rejected')
            ]
            ax.legend(handles=legend_patches, loc='lower center', ncol=2,
                      frameon=False, fontsize=10,
                      labelcolor='white', bbox_to_anchor=(0.5, -0.08))
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            card(f"""<div style="font-size:13px;font-weight:600;color:{PALETTE['text_muted']};
                letter-spacing:0.07em;text-transform:uppercase;margin-bottom:14px;">
                CIBIL Score Distribution</div>""", padding="18px 20px 0px 20px")
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="none")
            ax2.set_facecolor("none")
            for spine in ax2.spines.values():
                spine.set_color(PALETTE['border'])
            approved_scores = df[df['loan_status'] == 'Approved']['cibil_score']
            rejected_scores = df[df['loan_status'] == 'Rejected']['cibil_score']
            ax2.hist(approved_scores, bins=25, alpha=0.7, color=PALETTE['success'],
                     label='Approved', edgecolor='none')
            ax2.hist(rejected_scores, bins=25, alpha=0.7, color=PALETTE['danger'],
                     label='Rejected', edgecolor='none')
            ax2.set_xlabel('CIBIL Score', color=PALETTE['text_muted'], fontsize=11)
            ax2.set_ylabel('Count', color=PALETTE['text_muted'], fontsize=11)
            ax2.tick_params(colors=PALETTE['text_muted'])
            ax2.legend(frameon=False, labelcolor='white', fontsize=10)
            ax2.axvline(x=750, color=PALETTE['gold'], linestyle='--', linewidth=1.5,
                        label='Threshold 750')
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        with c3:
            card(f"""<div style="font-size:13px;font-weight:600;color:{PALETTE['text_muted']};
                letter-spacing:0.07em;text-transform:uppercase;margin-bottom:14px;">
                Income vs Approval</div>""", padding="18px 20px 0px 20px")
            fig3, ax3 = plt.subplots(figsize=(5, 4), facecolor="none")
            ax3.set_facecolor("none")
            for spine in ax3.spines.values():
                spine.set_color(PALETTE['border'])
            groups = df.groupby('loan_status')['income_annum'].mean() / 1_00_000
            bars = ax3.bar(groups.index, groups.values,
                           color=[PALETTE['success'], PALETTE['danger']],
                           width=0.5, edgecolor='none', alpha=0.9)
            for bar, val in zip(bars, groups.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'₹{val:.1f}L', ha='center', color='white', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Avg Income (Lakhs)', color=PALETTE['text_muted'], fontsize=11)
            ax3.tick_params(colors=PALETTE['text_muted'])
            st.pyplot(fig3, use_container_width=True)
            plt.close()
    else:
        st.warning("Dataset not found. Please ensure `loan_approval_dataset (1).csv` is in the working directory.")

# ────────────────────────────────────
# TAB 2 — LOAN ASSESSMENT
# ────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="font-size:22px;font-weight:800;color:{PALETTE['text_primary']};">
            Loan Eligibility Assessment
        </div>
        <div style="color:{PALETTE['text_muted']};font-size:14px;margin-top:4px;">
            Fill in the applicant details below to get an instant AI-driven decision.
        </div>
    </div>
    """, unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        # ── Section A: Personal
        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            A · Personal Information</div>""", unsafe_allow_html=True)

        pc1, pc2 = st.columns(2)
        with pc1:
            dependents = st.number_input("No. of Dependents", min_value=0, max_value=10, value=0)
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        with pc2:
            self_employed = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
            income = st.number_input("Annual Income (₹)", min_value=0, step=50000, value=500000)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section B: Loan Details
        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            B · Loan Details</div>""", unsafe_allow_html=True)

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=100000, value=1000000)
        with lc2:
            loan_term = st.number_input(
    "Term (Years)",
    min_value=2,
    max_value=20,
    value=10,
    step=1
)
        with lc3:
            cibil = st.slider("CIBIL Score", 300, 900, 700)

        cibil_color = PALETTE['danger'] if cibil < 600 else (PALETTE['gold'] if cibil < 750 else PALETTE['success'])
        cibil_label = "Poor" if cibil < 600 else ("Fair" if cibil < 750 else "Excellent")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-top:-6px;margin-bottom:14px;">
            <div style="height:6px;flex:1;background:{PALETTE['steel']};border-radius:4px;overflow:hidden;">
                <div style="height:100%;width:{(cibil-300)/600*100:.0f}%;
                    background:linear-gradient(90deg,{PALETTE['danger']},{PALETTE['gold']},{PALETTE['success']});
                    border-radius:4px;"></div>
            </div>
            <span style="font-size:12px;font-weight:700;color:{cibil_color};min-width:60px;">{cibil} · {cibil_label}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section C: Assets
        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            C · Asset Breakdown</div>""", unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        with ac1:
            residential = st.number_input("Residential Assets (₹)", min_value=0, step=100000)
            commercial = st.number_input("Commercial Assets (₹)", min_value=0, step=100000)
        with ac2:
            luxury = st.number_input("Luxury Assets (₹)", min_value=0, step=100000)
            bank = st.number_input("Bank / Liquid Assets (₹)", min_value=0, step=10000)

        total_assets = residential + commercial + luxury + bank
        st.markdown(f"""
        <div style="
            background:{PALETTE['steel']};
            border:1px solid {PALETTE['border']};
            border-radius:10px;
            padding:12px 18px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-top:6px;
            margin-bottom:20px;
        ">
            <span style="color:{PALETTE['text_muted']};font-size:13px;font-weight:600;">Total Assets</span>
            <span style="color:{PALETTE['text_primary']};font-size:17px;font-weight:800;
                font-family:'DM Mono',monospace;">{fmt_inr(total_assets)}</span>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🚀  Run Credit Assessment", use_container_width=True)

    # ── RESULT PANEL
    with result_col:
        if predict_btn:
            if not model_loaded:
                st.error("Model file not found. Please ensure `trained_model.pkl` and `feature_columns.pkl` are present.")
            else:
                education_encoded = 0 if education == "Graduate" else 1
                self_employed_encoded = 1 if self_employed == "Self-Employed" else 0

                user_input = pd.DataFrame([[
                    dependents, education_encoded, self_employed_encoded,
                    income, loan_amount, loan_term, cibil,
                    residential, commercial, luxury, bank, total_assets
                ]], columns=columns)

                probs = model.predict_proba(user_input)[0]
                approval_prob = probs[0]
                rejection_prob = probs[1]
                approved_flag = approval_prob > 0.5
                if (cibil < 600 or income < 500000 or loan_amount > income * 3):
                        approved_flag = False

                # Decision card
                if approved_flag:
                    decision_bg = f"linear-gradient(135deg, #064E3B, #065F46)"
                    decision_border = PALETTE['success']
                    decision_icon = "✅"
                    decision_text = "APPROVED"
                    conf_val = approval_prob
                else:
                    decision_bg = f"linear-gradient(135deg, #450A0A, #7F1D1D)"
                    decision_border = PALETTE['danger']
                    decision_icon = "❌"
                    decision_text = "DECLINED"
                    conf_val = rejection_prob

                st.markdown(f"""
                <div style="
                    background:{decision_bg};
                    border:1.5px solid {decision_border};
                    border-radius:18px;
                    padding:30px 24px;
                    text-align:center;
                    box-shadow:0 8px 32px {decision_border}33;
                    margin-bottom:20px;
                ">
                    <div style="font-size:40px;margin-bottom:8px;">{decision_icon}</div>
                    <div style="font-size:28px;font-weight:800;color:white;
                        letter-spacing:0.04em;">{decision_text}</div>
                    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:4px;
                        text-transform:uppercase;letter-spacing:0.08em;">
                        Confidence score
                    </div>
                    <div style="font-size:42px;font-weight:800;color:white;
                        font-family:'DM Mono',monospace;margin-top:4px;">
                        {conf_val*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar
                st.markdown(f"""
                <div style="margin-bottom:20px;">
                    <div style="display:flex;justify-content:space-between;
                        font-size:12px;color:{PALETTE['text_muted']};margin-bottom:6px;">
                        <span>Approval Probability</span>
                        <span style="font-family:'DM Mono',monospace;font-weight:700;
                            color:{PALETTE['success']};">{approval_prob*100:.1f}%</span>
                    </div>
                    <div style="height:10px;background:{PALETTE['steel']};border-radius:6px;overflow:hidden;">
                        <div style="height:100%;width:{approval_prob*100:.1f}%;
                            background:linear-gradient(90deg,{PALETTE['danger']},{PALETTE['success']});
                            border-radius:6px;transition:width 0.6s ease;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Quick score card
                dti = (loan_amount / income * 100) if income > 0 else 0
                ltv = (loan_amount / total_assets * 100) if total_assets > 0 else 999
                score_items = [
                    ("CIBIL Score", f"{cibil}", cibil_color, cibil_label),
                    ("Debt-to-Income", f"{dti:.1f}%", PALETTE['success'] if dti < 40 else PALETTE['danger'], "Healthy" if dti < 40 else "High"),
                    ("Loan-to-Asset", f"{min(ltv,999):.1f}%", PALETTE['success'] if ltv < 80 else PALETTE['danger'], "Safe" if ltv < 80 else "Risky"),
                    ("Annual Income", fmt_inr(income), PALETTE['success'] if income >= 500000 else PALETTE['danger'], "Adequate" if income >= 500000 else "Low"),
                ]
                st.markdown(f"""
                <div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
                    Risk Indicators</div>""", unsafe_allow_html=True)
                for label_s, val_s, color_s, tag_s in score_items:
                    st.markdown(f"""
                    <div style="
                        display:flex;align-items:center;justify-content:space-between;
                        background:{PALETTE['steel']};border-radius:10px;
                        padding:10px 16px;margin-bottom:8px;
                        border-left:3px solid {color_s};
                    ">
                        <span style="color:{PALETTE['text_muted']};font-size:13px;">{label_s}</span>
                        <div style="display:flex;gap:10px;align-items:center;">
                            <span style="font-family:'DM Mono',monospace;font-size:14px;
                                font-weight:700;color:white;">{val_s}</span>
                            {badge(tag_s, color_s)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Suggested loans / Rejection reasons
                if approved_flag:
                    suggestions = []
                    if loan_amount <= 2_000_000:
                        suggestions.append(("👤 Personal Loan", "Quick disbursal, minimal docs"))
                    if loan_amount > 2_000_000:
                        suggestions.append(("🏠 Home Loan", "Low interest, long tenure"))
                    if self_employed_encoded:
                        suggestions.append(("🏢 Business Loan", "Tailored for entrepreneurs"))
                    if suggestions:
                        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
                            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
                            Suitable Products</div>""", unsafe_allow_html=True)
                        for name, desc in suggestions:
                            st.markdown(f"""
                            <div style="
                                background:{PALETTE['navy_card']};
                                border:1px solid {PALETTE['success']}44;
                                border-radius:10px;padding:12px 16px;margin-bottom:8px;
                                display:flex;justify-content:space-between;align-items:center;
                            ">
                                <div>
                                    <div style="color:white;font-weight:600;font-size:14px;">{name}</div>
                                    <div style="color:{PALETTE['text_muted']};font-size:12px;">{desc}</div>
                                </div>
                                <span style="color:{PALETTE['success']};font-size:18px;">→</span>
                            </div>""", unsafe_allow_html=True)
                else:
                    reasons = []
                    if cibil < 600:
                        reasons.append(("📉 Low CIBIL Score", f"Score {cibil} is below the 600 threshold"))
                    if income < 500000:
                        reasons.append(("💸 Insufficient Income", f"Annual income {fmt_inr(income)} below ₹5L"))
                    if dti > 50:
                        reasons.append(("⚠️ High Debt-to-Income", f"DTI of {dti:.1f}% exceeds 50% limit"))
                    if total_assets < loan_amount:
                        reasons.append(("🏦 Asset Shortfall", f"Assets {fmt_inr(total_assets)} < Loan {fmt_inr(loan_amount)}"))
                    if reasons:
                        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['danger']};
                            letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;">
                            Decline Reasons</div>""", unsafe_allow_html=True)
                        for title_r, detail_r in reasons:
                            st.markdown(f"""
                            <div style="
                                background:{PALETTE['navy_card']};
                                border:1px solid {PALETTE['danger']}44;
                                border-radius:10px;padding:12px 16px;margin-bottom:8px;
                            ">
                                <div style="color:white;font-weight:600;font-size:14px;">{title_r}</div>
                                <div style="color:{PALETTE['text_muted']};font-size:12px;">{detail_r}</div>
                            </div>""", unsafe_allow_html=True)

        else:
            # Placeholder state
            st.markdown(f"""
            <div style="
                border:2px dashed {PALETTE['border']};
                border-radius:18px;
                padding:60px 30px;
                text-align:center;
                color:{PALETTE['text_muted']};
            ">
                <div style="font-size:48px;margin-bottom:16px;">🎯</div>
                <div style="font-size:18px;font-weight:700;color:{PALETTE['text_primary']};
                    margin-bottom:8px;">Assessment Ready</div>
                <div style="font-size:14px;line-height:1.6;">
                    Complete the applicant form on the left<br>and click <strong style="color:white;">Run Credit Assessment</strong><br>
                    to get an instant AI-powered decision.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────────
# TAB 3 — DATASET ANALYTICS
# ────────────────────────────────────
with tab3:
    if df is not None:
        st.markdown(f"""
        <div style="margin-bottom:24px;">
            <div style="font-size:22px;font-weight:800;color:{PALETTE['text_primary']};">
                Dataset Analytics
            </div>
            <div style="color:{PALETTE['text_muted']};font-size:14px;margin-top:4px;">
                Exploring {len(df):,} loan applications · {df['loan_status'].value_counts().get('Approved', 0):,} approved · {df['loan_status'].value_counts().get('Rejected', 0):,} rejected
            </div>
        </div>
        """, unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)

        with r1c1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
            ax.set_facecolor("none")
            for spine in ax.spines.values(): spine.set_color(PALETTE['border'])
            education_approval = df.groupby(['education', 'loan_status']).size().unstack(fill_value=0)
            x = np.arange(len(education_approval.index))
            width = 0.35
            if 'Approved' in education_approval.columns:
                ax.bar(x - width/2, education_approval['Approved'], width, label='Approved',
                       color=PALETTE['success'], alpha=0.85, edgecolor='none')
            if 'Rejected' in education_approval.columns:
                ax.bar(x + width/2, education_approval['Rejected'], width, label='Rejected',
                       color=PALETTE['danger'], alpha=0.85, edgecolor='none')
            ax.set_xticks(x)
            ax.set_xticklabels(education_approval.index, color=PALETTE['text_muted'])
            ax.tick_params(colors=PALETTE['text_muted'])
            ax.set_title('Education vs Loan Status', color='white', fontsize=13, fontweight='bold', pad=12)
            ax.legend(frameon=False, labelcolor='white')
            st.pyplot(fig, use_container_width=True); plt.close()

        with r1c2:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
            ax.set_facecolor("none")
            for spine in ax.spines.values(): spine.set_color(PALETTE['border'])
            dep_approval = df.groupby('no_of_dependents')['loan_status'].apply(
                lambda x: (x == 'Approved').mean() * 100
            )
            ax.bar(dep_approval.index, dep_approval.values,
                   color=PALETTE['accent'], alpha=0.85, edgecolor='none', width=0.6)
            ax.axhline(y=50, color=PALETTE['gold'], linestyle='--', linewidth=1.5)
            ax.set_xlabel('Number of Dependents', color=PALETTE['text_muted'])
            ax.set_ylabel('Approval Rate (%)', color=PALETTE['text_muted'])
            ax.tick_params(colors=PALETTE['text_muted'])
            ax.set_title('Approval Rate by Dependents', color='white', fontsize=13, fontweight='bold', pad=12)
            st.pyplot(fig, use_container_width=True); plt.close()

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
            ax.set_facecolor("none")
            for spine in ax.spines.values(): spine.set_color(PALETTE['border'])
            for status, color in [('Approved', PALETTE['success']), ('Rejected', PALETTE['danger'])]:
                subset = df[df['loan_status'] == status]['loan_amount'] / 1_00_000
                ax.hist(subset, bins=30, alpha=0.65, color=color, label=status, edgecolor='none')
            ax.set_xlabel('Loan Amount (Lakhs ₹)', color=PALETTE['text_muted'])
            ax.set_ylabel('Count', color=PALETTE['text_muted'])
            ax.tick_params(colors=PALETTE['text_muted'])
            ax.set_title('Loan Amount Distribution', color='white', fontsize=13, fontweight='bold', pad=12)
            ax.legend(frameon=False, labelcolor='white')
            st.pyplot(fig, use_container_width=True); plt.close()

        with r2c2:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="none")
            ax.set_facecolor("none")
            for spine in ax.spines.values(): spine.set_color(PALETTE['border'])
            for status, color in [('Approved', PALETTE['success']), ('Rejected', PALETTE['danger'])]:
                subset = df[df['loan_status'] == status]['total_assets'] / 1_00_000
                ax.hist(subset, bins=30, alpha=0.65, color=color, label=status, edgecolor='none')
            ax.set_xlabel('Total Assets (Lakhs ₹)', color=PALETTE['text_muted'])
            ax.set_ylabel('Count', color=PALETTE['text_muted'])
            ax.tick_params(colors=PALETTE['text_muted'])
            ax.set_title('Total Asset Distribution', color='white', fontsize=13, fontweight='bold', pad=12)
            ax.legend(frameon=False, labelcolor='white')
            st.pyplot(fig, use_container_width=True); plt.close()

        # Raw data expander
        with st.expander("🗃  View Raw Dataset"):
            st.dataframe(
                df.head(100).style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', PALETTE['steel']), ('color', 'white')]},
                    {'selector': 'td', 'props': [('background-color', PALETTE['navy_card']), ('color', PALETTE['text_primary'])]},
                ]),
                use_container_width=True
            )
    else:
        st.warning("Dataset not available.")

# ────────────────────────────────────
# TAB 4 — FEATURE INSIGHTS
# ────────────────────────────────────
with tab4:
    if df is not None:
        st.markdown(f"""
        <div style="margin-bottom:24px;">
            <div style="font-size:22px;font-weight:800;color:{PALETTE['text_primary']};">
                Feature Insights
            </div>
            <div style="color:{PALETTE['text_muted']};font-size:14px;margin-top:4px;">
                Correlation analysis and variable relationships across the dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

        fc1, fc2 = st.columns([1.3, 1])

        with fc1:
            numeric_cols = ['no_of_dependents', 'income_annum', 'loan_amount',
                            'loan_term', 'cibil_score', 'total_assets']
            available = [c for c in numeric_cols if c in df.columns]
            corr = df[available].corr()

            fig, ax = plt.subplots(figsize=(7, 5), facecolor="none")
            ax.set_facecolor("none")
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(
                corr, mask=mask, cmap=cmap, annot=True, fmt=".2f", ax=ax,
                annot_kws={"size": 10, "color": "white"},
                linewidths=0.5, linecolor=PALETTE['navy'],
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title('Feature Correlation Matrix', color='white', fontsize=13,
                         fontweight='bold', pad=14)
            ax.tick_params(colors=PALETTE['text_muted'])
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig, use_container_width=True); plt.close()

        with fc2:
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
            ax.set_facecolor("none")
            for spine in ax.spines.values(): spine.set_color(PALETTE['border'])
            for status, color, marker in [
                ('Approved', PALETTE['success'], 'o'),
                ('Rejected', PALETTE['danger'], 'x')
            ]:
                subset = df[df['loan_status'] == status].sample(min(300, len(df)))
                ax.scatter(
                    subset['cibil_score'],
                    subset['income_annum'] / 1_00_000,
                    alpha=0.4, color=color, label=status,
                    s=18, marker=marker
                )
            ax.set_xlabel('CIBIL Score', color=PALETTE['text_muted'])
            ax.set_ylabel('Annual Income (Lakhs)', color=PALETTE['text_muted'])
            ax.tick_params(colors=PALETTE['text_muted'])
            ax.set_title('CIBIL Score vs Income', color='white', fontsize=13,
                         fontweight='bold', pad=14)
            ax.legend(frameon=False, labelcolor='white')
            st.pyplot(fig, use_container_width=True); plt.close()

        # Summary stats
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:11px;font-weight:700;color:{PALETTE['accent']};
            letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
            Summary Statistics</div>""", unsafe_allow_html=True)
        summary = df[available].describe().round(2)
        st.dataframe(summary, use_container_width=True)
    else:
        st.warning("Dataset not available.")

# ====================================
# FOOTER
# ====================================
st.markdown(f"""
<div style="
    text-align:center;
    padding:32px 0 16px 0;
    color:{PALETTE['text_muted']};
    font-size:13px;
    border-top:1px solid {PALETTE['border']};
    margin-top:40px;
">
    LoanIQ · AI Credit Assessment Platform · Built with Streamlit & scikit-learn
    <br><span style="font-size:11px;opacity:0.5;">For demonstration purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)