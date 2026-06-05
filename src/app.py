# MULTI-PAGE STREAMLIT DASHBOARD - MAIN ENTRY POINT

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from io import BytesIO

from decision_support import enrich_decision_columns, decision_export_columns

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Retention Hub",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# SHARED FUNCTIONS
# =====================================================

@st.cache_resource
def load_model():
    """Load trained RF model for next product category prediction."""
    try:
        model = joblib.load("models/next_product_category_rf_model.joblib")
        encoder = joblib.load("models/category_encoder.joblib")
        return model, encoder
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None


@st.cache_resource
def load_churn_model():
    """Load trained churn/repurchase model."""
    try:
        return joblib.load("models/best_churn_model.joblib")
    except Exception as e:
        st.error(f"Churn model load error: {e}")
        return None


@st.cache_data
def load_data(uploaded_file):
    """Load, clean, and enrich customer dataset."""
    if uploaded_file is None:
        return None

    df = pd.read_csv(uploaded_file)
    df['user_id'] = df['user_id'].astype(str).fillna("UNKNOWN")

    # Numeric columns
    numeric_cols = [
        'age', 'total_view', 'total_click', 'total_cart', 'total_wishlist',
        'click_through_rate', 'cart_rate', 'unique_brands',
        'avg_price', 'avg_discount', 'avg_purchase_value', 'avg_rating',
        'user_total_categories', 'category_share',
        'view_to_click_ratio', 'cart_to_click_ratio', 'wishlist_to_cart_ratio',
        'active_engagement_ratio', 'category_commitment_score',
        'exploration_score', 'engagement_depth_score', 'discount_sensitivity',
        'brand_loyalty_score', 'click_engagement_rate', 'value_per_click',
        'price_to_value_ratio', 'category_breadth_score', 'rating_concentration'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Customer name
    df['customer_name'] = "Customer " + df['user_id']

    # Repurchase status
    if 'repurchase_pred' in df.columns:
        df['repurchase_pred'] = pd.to_numeric(df['repurchase_pred'], errors='coerce').fillna(0).astype(int)
        df['qualified_status'] = np.where(df['repurchase_pred'] == 1, 'Qualified', 'Not Qualified')
    elif 'repurchase' in df.columns:
        df['repurchase'] = pd.to_numeric(df['repurchase'], errors='coerce').fillna(0).astype(int)
        df['qualified_status'] = np.where(df['repurchase'] == 1, 'Qualified', 'Not Qualified')
    else:
        df['qualified_status'] = 'Unknown'

    # IC-based segmentation
    ic_features = [
        'total_view', 'total_click', 'total_cart', 'total_wishlist',
        'click_through_rate', 'cart_rate', 'view_to_click_ratio',
        'cart_to_click_ratio', 'wishlist_to_cart_ratio', 'active_engagement_ratio'
    ]

    for i, col in enumerate(ic_features, start=1):
        if col in df.columns:
            median_val = df[col].median()
            df[f'ic_{i}'] = (df[col] >= median_val).astype(int)
        else:
            df[f'ic_{i}'] = 0

    ic_cols = [f'ic_{i}' for i in range(1, len(ic_features) + 1)]
    pass_rates = {col: df[col].mean() for col in ic_cols}
    ic_weights = {col: -np.log(max(rate, 1e-4)) for col, rate in pass_rates.items()}
    df['ic_full_score'] = sum(df[col] * ic_weights[col] for col in ic_cols)
    df['pass_count'] = df[ic_cols].sum(axis=1)

    df['behavior_segment'] = np.select(
        [
            df['pass_count'] >= 8,
            df['pass_count'].between(6, 7),
            df['pass_count'].between(4, 5)
        ],
        ['Very High Potential', 'High Potential', 'Medium Potential'],
        default='Low Potential'
    )

    return df


def find_purchase_label_column(columns):
    """Detect repurchase label column."""
    lower_cols = [c.lower() for c in columns]
    if 'repurchase_pred' in lower_cols:
        return columns[lower_cols.index('repurchase_pred')]
    if 'repurchase' in lower_cols:
        return columns[lower_cols.index('repurchase')]
    return None


def build_excel_bytes(df):
    """Build an Excel file for business users. Return None if engine is missing."""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="retention_actions")
        return output.getvalue()
    except ImportError:
        return None


def safe_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value):
    return f"{safe_number(value) * 100:.1f}%"


def customer_category_scope(df, customer):
    if "product_category" not in df.columns:
        return df
    category = customer.get("product_category")
    return df[df["product_category"].astype(str) == str(category)]


def build_single_customer_report(customer):
    report_fields = {
        "User ID": customer.get("user_id", "N/A"),
        "Category": customer.get("product_category", "N/A"),
        "Repurchase Probability": format_percent(customer.get("repurchase_probability", 0)),
        "Risk Level": customer.get("risk_level", "N/A"),
        "Priority": customer.get("priority_level", "N/A"),
        "Behavior Segment": customer.get("behavior_segment", "N/A"),
        "Recommended Action": customer.get("recommended_action", "N/A"),
        "Main Reason": customer.get("decision_reason", "N/A"),
        "Average Purchase Value": safe_number(customer.get("avg_purchase_value", 0)),
        "Engagement Depth": safe_number(customer.get("engagement_depth_score", 0)),
        "Brand Loyalty": safe_number(customer.get("brand_loyalty_score", 0)),
        "Cart Rate": safe_number(customer.get("cart_rate", 0)),
        "Discount Sensitivity": safe_number(customer.get("discount_sensitivity", 0)),
    }
    return pd.DataFrame(report_fields.items(), columns=["Field", "Value"])


def build_model_input(customer, model, aliases=None):
    aliases = aliases or {}
    values = {}

    for feature in getattr(model, "feature_names_in_", []):
        if feature in customer.index:
            values[feature] = safe_number(customer.get(feature, 0))
            continue

        alias_value = 0
        for alias in aliases.get(feature, []):
            if alias in customer.index:
                alias_value = safe_number(customer.get(alias, 0))
                break
        values[feature] = alias_value

    return pd.DataFrame([values], columns=model.feature_names_in_)


def predict_churn_customer(customer, churn_model):
    aliases = {
        "Monetary": ["Monetary", "avg_purchase_value", "value_per_click", "avg_price"],
        "Frequency": ["Frequency", "total_cart", "total_click", "total_view"],
        "Recency": ["Recency", "recency", "days_since_last_purchase"],
        "Tenure": ["Tenure", "tenure", "user_total_categories"],
    }
    input_data = build_model_input(customer, churn_model, aliases)
    pred = int(churn_model.predict(input_data)[0])
    probability = None
    if hasattr(churn_model, "predict_proba"):
        probs = churn_model.predict_proba(input_data)[0]
        class_list = list(churn_model.classes_)
        positive_index = class_list.index(1) if 1 in class_list else int(np.argmax(probs))
        probability = float(probs[positive_index])
    return pred, probability, input_data


def predict_next_categories(customer, category_model, category_encoder, top_n=3):
    input_data = build_model_input(
        customer,
        category_model,
        {
            "dominant_gender": ["dominant_gender"],
            "dominant_location": ["dominant_location"],
            "wishlist_to_view_ratio": ["wishlist_to_view_ratio", "wishlist_to_cart_ratio"],
        },
    )

    if hasattr(category_model, "predict_proba"):
        probs = category_model.predict_proba(input_data)[0]
        top_idx = np.argsort(probs)[-top_n:][::-1]
        categories = category_encoder.inverse_transform(top_idx)
        return pd.DataFrame(
            {
                "Predicted Category": categories,
                "Probability": probs[top_idx] * 100,
            }
        )

    pred = category_model.predict(input_data)
    category = category_encoder.inverse_transform(pred)[0]
    return pd.DataFrame({"Predicted Category": [category], "Probability": [np.nan]})


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'selected_customer_data' not in st.session_state:
    st.session_state.selected_customer_data = None
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None

# =====================================================
# MAIN PAGE: DASHBOARD & DATA UPLOAD
# =====================================================

st.title("📊 Customer Retention Hub")
st.markdown("Analyze customer behavior, predict repurchase patterns, and plan retention strategies.")

# Sidebar
st.sidebar.title("⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"], key="main_uploader")

if uploaded_file is None:
    st.sidebar.info("Upload a CSV file to get started.")
    st.info("👋 Welcome! Please upload a customer dataset (CSV format) to begin analysis.")
    st.stop()

# Load data
with st.spinner("Loading dataset..."):
    df = load_data(uploaded_file)
    df = enrich_decision_columns(df)
    st.session_state.df = df

if df is None or df.empty:
    st.error("Failed to load or parse the dataset.")
    st.stop()

# Display dataset info
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.write(f"**File:** {uploaded_file.name}")
st.sidebar.write(f"**Rows:** {len(df):,}")
st.sidebar.write(f"**Users:** {df['user_id'].nunique():,}")
if 'product_category' in df.columns:
    st.sidebar.write(f"**Categories:** {df['product_category'].nunique():,}")
st.sidebar.write(f"**Features:** {len(df.columns):,}")
st.sidebar.markdown("---")
st.sidebar.info("📌 Use the page menu (left) to explore profiles, manage customers, and plan actions.")

# Load model
model, encoder = load_model()
churn_model = load_churn_model()
st.session_state.model = model
st.session_state.encoder = encoder
st.session_state.churn_model = churn_model

purchase_column = find_purchase_label_column(df.columns.tolist())

# =====================================================
# MAIN DASHBOARD CONTENT
# =====================================================

st.markdown("---")
st.subheader("🎯 Business Overview")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Customers", df['user_id'].nunique())
with col2:
    if 'product_category' in df.columns:
        st.metric("Categories", df['product_category'].nunique())
    else:
        st.metric("Categories", "N/A")
with col3:
    avg_repurchase = df["repurchase_probability"].mean() * 100
    st.metric("Avg Repurchase", f"{avg_repurchase:.1f}%")
with col4:
    high_risk = int((df["risk_level"] == "High Risk").sum())
    st.metric("High Risk", high_risk)
with col5:
    p1_count = int((df["priority_level"] == "P1 - Save Now").sum())
    st.metric("P1 Actions", p1_count)

st.markdown("---")
st.subheader("🔴 Priority Action List")

priority_order = {
    "P1 - Save Now": 1,
    "P2 - Win Back": 2,
    "P2 - Nurture": 3,
    "P3 - Upsell": 4,
    "P4 - Monitor": 5,
}
action_df = df.copy()
action_df["_priority_rank"] = action_df["priority_level"].map(priority_order).fillna(99)
action_df = action_df.sort_values(
    ["_priority_rank", "repurchase_probability", "avg_purchase_value"],
    ascending=[True, True, False],
)

if action_df.empty:
    st.info("No at-risk customers detected.")
else:
    display_cols = [
        "user_id",
        "product_category",
        "repurchase_probability",
        "risk_level",
        "priority_level",
        "recommended_action",
        "decision_reason",
    ]
    display_cols = [col for col in display_cols if col in action_df.columns]

    top_actions = action_df.head(20)[display_cols].copy()
    rename_map = {
        "user_id": "User ID",
        "product_category": "Category",
        "repurchase_probability": "Repurchase Probability",
        "risk_level": "Risk",
        "priority_level": "Priority",
        "recommended_action": "Recommended Action",
        "decision_reason": "Main Reason",
    }
    top_actions = top_actions.rename(columns=rename_map)
    if "Repurchase Probability" in top_actions.columns:
        top_actions["Repurchase Probability"] = top_actions["Repurchase Probability"].map(lambda x: f"{x * 100:.1f}%")

    st.dataframe(top_actions, use_container_width=True, hide_index=True)

    export_cols = decision_export_columns(action_df)
    export_df = action_df[export_cols].copy()
    excel_bytes = build_excel_bytes(export_df)
    if excel_bytes is not None:
        st.download_button(
            "Export Excel Action List",
            data=excel_bytes,
            file_name="customer_retention_action_list.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.download_button(
            "Export CSV Action List",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="customer_retention_action_list.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")
st.subheader("📊 Decision Views")

tab_marketing, tab_sales, tab_manager = st.tabs(["Marketing", "Sales", "Manager"])

with tab_marketing:
    risk_summary = df["risk_level"].value_counts().reset_index()
    risk_summary.columns = ["Risk Level", "Customers"]
    fig = px.bar(risk_summary, x="Risk Level", y="Customers", color="Risk Level", title="Customers by Risk Level")
    st.plotly_chart(fig, use_container_width=True)

    campaign_summary = df["recommended_action"].value_counts().reset_index()
    campaign_summary.columns = ["Campaign Action", "Customers"]
    st.dataframe(campaign_summary, use_container_width=True, hide_index=True)

with tab_sales:
    sales_cols = [
        "user_id",
        "product_category",
        "avg_purchase_value",
        "repurchase_probability",
        "priority_level",
        "recommended_action",
    ]
    sales_cols = [col for col in sales_cols if col in df.columns]
    sales_df = df.sort_values(["avg_purchase_value", "repurchase_probability"], ascending=[False, False]).head(20)
    st.dataframe(sales_df[sales_cols], use_container_width=True, hide_index=True)

with tab_manager:
    mgr_col1, mgr_col2 = st.columns(2)
    with mgr_col1:
        priority_summary = df["priority_level"].value_counts().reset_index()
        priority_summary.columns = ["Priority", "Customers"]
        fig = px.pie(priority_summary, names="Priority", values="Customers", title="Workload by Priority")
        st.plotly_chart(fig, use_container_width=True)
    with mgr_col2:
        if "product_category" in df.columns:
            category_summary = (
                df.groupby("product_category", as_index=False)
                .agg(
                    customers=("user_id", "nunique"),
                    avg_repurchase=("repurchase_probability", "mean"),
                    avg_value=("avg_purchase_value", "mean"),
                )
                .sort_values("avg_repurchase", ascending=False)
            )
            category_summary["avg_repurchase"] = category_summary["avg_repurchase"] * 100
            fig = px.bar(
                category_summary.head(10),
                x="product_category",
                y="avg_repurchase",
                title="Top Categories by Expected Repurchase",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category column not available.")

st.markdown("---")
st.subheader("👤 Select Customer for Profile & Management")

# Customer selection
col_cat, col_user, col_btn = st.columns([2, 2, 1])

with col_cat:
    category_opts = ["All categories"]
    if 'product_category' in df.columns:
        category_opts += sorted(df['product_category'].dropna().astype(str).unique().tolist())
    
    selected_cat = st.selectbox("Category Filter", category_opts, index=0)

with col_user:
    if selected_cat == "All categories":
        user_opts = sorted(df['user_id'].astype(str).unique().tolist())
    else:
        user_opts = sorted(
            df[df['product_category'].astype(str) == selected_cat]['user_id'].astype(str).unique().tolist()
        )
    
    selected_usr = st.selectbox("User ID", ["Select user"] + user_opts, index=0)

with col_btn:
    if st.button("Load Customer", use_container_width=True):
        if selected_usr == "Select user":
            st.warning("Please select a valid user.")
        else:
            # Get customer data
            query = df[df['user_id'].astype(str) == str(selected_usr)]
            if selected_cat != "All categories" and 'product_category' in df.columns:
                query = query[query['product_category'].astype(str) == selected_cat]
            
            if query.empty:
                st.warning("Customer not found.")
            else:
                st.session_state.selected_user = selected_usr
                st.session_state.selected_customer_data = query.iloc[0].to_dict()
                st.session_state.selected_category = selected_cat if selected_cat != "All categories" else None
                st.success(f"✅ Loaded {selected_usr}. Visit other pages for detailed analysis.")

st.markdown("---")

if st.session_state.selected_customer_data:
    st.subheader("✨ Selected Customer Preview")
    cust = pd.Series(st.session_state.selected_customer_data)
    cust_scope = customer_category_scope(df, cust)
    category_name = cust.get("product_category", "N/A")
    category_avg_repurchase = cust_scope["repurchase_probability"].mean() if "repurchase_probability" in cust_scope else 0
    category_avg_value = cust_scope["avg_purchase_value"].mean() if "avg_purchase_value" in cust_scope else 0
    category_high_risk = int((cust_scope["risk_level"] == "High Risk").sum()) if "risk_level" in cust_scope else 0
    customer_expected_value = safe_number(cust.get("avg_purchase_value", 0)) * safe_number(cust.get("repurchase_probability", 0))
    
    pv_col1, pv_col2, pv_col3, pv_col4 = st.columns(4)
    with pv_col1:
        st.metric("User ID", st.session_state.selected_user)
    with pv_col2:
        st.metric("Category", cust.get('product_category', 'N/A'))
    with pv_col3:
        st.metric("Repurchase Probability", format_percent(cust.get("repurchase_probability", 0)))
    with pv_col4:
        st.metric("Risk / Priority", f"{cust.get('risk_level', 'N/A')} | {cust.get('priority_level', 'N/A')}")

    st.markdown("---")
    st.subheader("🔮 Predict Selected Customer")
    st.write(
        f"Run prediction for user `{st.session_state.selected_user}` in category `{category_name}`."
    )

    if st.button("Run Predict", type="primary", use_container_width=True):
        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.markdown("**Repurchase / Churn Prediction**")
            if churn_model is None:
                st.warning("Churn model is not available.")
            else:
                try:
                    churn_pred, churn_probability, churn_input = predict_churn_customer(cust, churn_model)
                    label = "Will Repurchase / Return" if churn_pred == 1 else "At Risk / May Not Return"
                    if churn_probability is None:
                        st.metric("Prediction", label)
                    else:
                        st.metric("Prediction", label, f"{churn_probability * 100:.1f}%")
                    st.caption("Model input used for this prediction:")
                    st.dataframe(churn_input, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Churn prediction failed: {e}")

        with pred_col2:
            st.markdown("**Next Product Category Prediction**")
            if model is None or encoder is None:
                st.warning("Category model or encoder is not available.")
            else:
                try:
                    category_predictions = predict_next_categories(cust, model, encoder)
                    display_predictions = category_predictions.copy()
                    if "Probability" in display_predictions.columns:
                        display_predictions["Probability"] = display_predictions["Probability"].map(
                            lambda x: "N/A" if pd.isna(x) else f"{x:.1f}%"
                        )
                    st.dataframe(display_predictions, use_container_width=True, hide_index=True)
                    chart_data = category_predictions.dropna(subset=["Probability"])
                    if not chart_data.empty:
                        fig = px.bar(
                            chart_data,
                            x="Predicted Category",
                            y="Probability",
                            title="Top Predicted Categories",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Category prediction failed: {e}")

    st.markdown("---")
    st.subheader("🧾 Detailed Customer Report by Role")
    st.caption("This report is generated from the selected category and user_id, not from a generic business description.")

    report_col1, report_col2, report_col3, report_col4 = st.columns(4)
    with report_col1:
        st.metric("Selected Category", category_name)
    with report_col2:
        st.metric("Customer Expected Value", f"{customer_expected_value:,.0f}")
    with report_col3:
        st.metric("Category Avg Repurchase", f"{category_avg_repurchase * 100:.1f}%")
    with report_col4:
        st.metric("High-Risk Customers in Category", category_high_risk)

    role_marketing, role_sales, role_manager, role_analyst = st.tabs(
        ["Marketing", "Sales", "Manager", "Data Analyst"]
    )

    with role_marketing:
        st.markdown("**Decision question:** Should this customer receive a retention campaign?")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Risk Level", cust.get("risk_level", "N/A"))
        with m2:
            st.metric("Recommended Campaign", cust.get("recommended_action", "N/A"))
        with m3:
            st.metric("Discount Sensitivity", f"{safe_number(cust.get('discount_sensitivity', 0)):.2f}")

        st.write(f"**Why:** {cust.get('decision_reason', 'N/A')}")
        st.write(
            f"**Action:** Send campaign for `{category_name}` to user `{st.session_state.selected_user}` "
            f"with priority `{cust.get('priority_level', 'N/A')}`."
        )

        marketing_table = pd.DataFrame(
            [
                ["Campaign Target", st.session_state.selected_user],
                ["Category", category_name],
                ["Offer/Message", cust.get("recommended_action", "N/A")],
                ["Reason to Personalize", cust.get("decision_reason", "N/A")],
                ["Success Metric", "Repurchase in selected category"],
            ],
            columns=["Item", "Value"],
        )
        st.dataframe(marketing_table, use_container_width=True, hide_index=True)

    with role_sales:
        st.markdown("**Decision question:** Should Sales contact this customer before others?")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Call Priority", cust.get("priority_level", "N/A"))
        with s2:
            st.metric("Avg Purchase Value", f"{safe_number(cust.get('avg_purchase_value', 0)):,.0f}")
        with s3:
            st.metric("Cart Rate", f"{safe_number(cust.get('cart_rate', 0)):.2f}")
        with s4:
            st.metric("Wishlist", int(safe_number(cust.get("total_wishlist", 0))))

        st.write(
            f"**Sales next step:** Contact user `{st.session_state.selected_user}` about `{category_name}` "
            f"and use the recommendation: {cust.get('recommended_action', 'N/A')}."
        )
        sales_signals = pd.DataFrame(
            [
                ["Total Views", safe_number(cust.get("total_view", 0))],
                ["Total Clicks", safe_number(cust.get("total_click", 0))],
                ["Total Cart", safe_number(cust.get("total_cart", 0))],
                ["Value per Click", safe_number(cust.get("value_per_click", 0))],
                ["Brand Loyalty", safe_number(cust.get("brand_loyalty_score", 0))],
            ],
            columns=["Sales Signal", "Value"],
        )
        st.dataframe(sales_signals, use_container_width=True, hide_index=True)

    with role_manager:
        st.markdown("**Decision question:** How does this customer affect category retention and workload?")
        manager_metrics = pd.DataFrame(
            {
                "Metric": ["Selected Customer", f"Category Avg: {category_name}"],
                "Repurchase Probability": [
                    safe_number(cust.get("repurchase_probability", 0)) * 100,
                    category_avg_repurchase * 100,
                ],
                "Avg Purchase Value": [
                    safe_number(cust.get("avg_purchase_value", 0)),
                    category_avg_value,
                ],
            }
        )
        fig = px.bar(
            manager_metrics,
            x="Metric",
            y="Repurchase Probability",
            title="Customer vs Category Repurchase Probability",
            text_auto=".1f",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"**Manager interpretation:** This customer is `{cust.get('risk_level', 'N/A')}` "
            f"with expected value `{customer_expected_value:,.0f}`. "
            f"The selected category currently has `{len(cust_scope):,}` rows and `{category_high_risk:,}` high-risk customers."
        )
        st.dataframe(manager_metrics, use_container_width=True, hide_index=True)

    with role_analyst:
        st.markdown("**Decision question:** Which input signals explain this customer's result?")
        analyst_features = [
            "repurchase_probability",
            "risk_level",
            "priority_level",
            "ic_full_score",
            "pass_count",
            "engagement_depth_score",
            "click_through_rate",
            "cart_rate",
            "active_engagement_ratio",
            "brand_loyalty_score",
            "discount_sensitivity",
            "avg_purchase_value",
            "category_share",
            "exploration_score",
        ]
        analyst_features = [col for col in analyst_features if col in cust.index]
        analyst_table = pd.DataFrame(
            [{"Feature": col, "Value": cust.get(col)} for col in analyst_features]
        )
        st.dataframe(analyst_table, use_container_width=True, hide_index=True)
        st.info(
            "Use this view for model interpretation and feature checking. "
            "Marketing/Sales users should rely on probability, risk, priority, reason, and action."
        )

    customer_report_df = build_single_customer_report(cust)
    customer_excel = build_excel_bytes(customer_report_df)
    if customer_excel is not None:
        st.download_button(
            "Export This Customer Report",
            data=customer_excel,
            file_name=f"customer_{st.session_state.selected_user}_{category_name}_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.download_button(
            "Export This Customer Report",
            data=customer_report_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"customer_{st.session_state.selected_user}_{category_name}_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    if st.button("Clear Selection", use_container_width=True):
        st.session_state.selected_user = None
        st.session_state.selected_customer_data = None
        st.session_state.selected_category = None
        st.rerun()

st.markdown("---")
st.subheader("📖 Navigation Guide")
st.write("""
- **Customer Profile**: View detailed customer insights, behavior metrics, and product predictions
- **Customer Management**: Plan retention actions and marketing campaigns
- **Segmentation**: Analyze customer groups by behavior, value, and engagement
- **Action Center**: Get AI-recommended campaigns and action tasks
- **Analytics**: Explore trends, correlations, and customer lifetime value
""")
