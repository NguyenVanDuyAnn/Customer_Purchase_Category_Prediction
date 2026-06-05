# UPDATED ENTERPRISE STREAMLIT DASHBOARD


# =====================================================
# IMPORT LIBRARIES
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Product Category Prediction",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #f3f4f6;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #111827;
}

.customer-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.customer-header {
    background: linear-gradient(
        135deg,
        #111827,
        #1f2937
    );

    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.st-key-load_customer_button button,
.st-key-load_customer_button button:hover,
.st-key-load_customer_button button:focus {
    background-color: #dc3545 !important;
    border-color: #dc3545 !important;
    color: white !important;
}

.st-key-load_customer_button button:active {
    background-color: #b91c1c !important;
    border-color: #b91c1c !important;
    color: white !important;
}

.st-key-predict_button button,
.st-key-predict_button button:hover,
.st-key-predict_button button:focus {
    background-color: #16a34a !important;
    border-color: #16a34a !important;
    color: white !important;
}

.st-key-predict_button button:active {
    background-color: #15803d !important;
    border-color: #15803d !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            "models/next_product_category_rf_model.joblib"
        )

        encoder = joblib.load(
            "models/category_encoder.joblib"
        )

        return model, encoder

    except Exception as e:

        st.error(f"Cannot load model: {e}")

        return None, None

model, category_encoder = load_model()

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data(uploaded_file=None):

    # =========================
    # LOAD DATA
    # =========================
    if uploaded_file is None:
        return None

    df = pd.read_csv(uploaded_file)

    # =========================
    # BASIC CLEAN
    # =========================
    df['user_id'] = df['user_id'].astype(str).fillna("UNKNOWN")

    # =========================
    # NUMERIC SAFE CAST
    # =========================
    numeric_cols = [
        'age',
        'total_view','total_click','total_cart','total_wishlist',
        'click_through_rate','cart_rate',
        'unique_brands',
        'avg_price','avg_discount','avg_purchase_value','avg_rating',
        'user_total_categories','category_share',
        'view_to_click_ratio','cart_to_click_ratio','wishlist_to_cart_ratio',
        'active_engagement_ratio','category_commitment_score',
        'exploration_score','engagement_depth_score','discount_sensitivity',
        'brand_loyalty_score','wishlist_to_cart_ratio',
        'click_engagement_rate','value_per_click','price_to_value_ratio',
        'category_breadth_score','rating_concentration'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # =========================
    # OPTIONAL DEMO LABELS (SAFE)
    # =========================
    df['customer_name'] = "Customer " + df['user_id']

    # =========================
    # USE EXISTING PURCHASE LABEL FROM DATA
    # =========================
    if 'repurchase_pred' in df.columns:
        df['repurchase_pred'] = pd.to_numeric(df['repurchase_pred'], errors='coerce').fillna(0).astype(int)
        df['qualified_status'] = np.where(df['repurchase_pred'] == 1, 'Qualified', 'Not Qualified')
    elif 'repurchase' in df.columns:
        df['repurchase'] = pd.to_numeric(df['repurchase'], errors='coerce').fillna(0).astype(int)
        df['qualified_status'] = np.where(df['repurchase'] == 1, 'Qualified', 'Not Qualified')
    else:
        df['qualified_status'] = 'Unknown'

    return df


def find_purchase_label_column(columns):
    lower_cols = [c.lower() for c in columns]
    if 'repurchase_pred' in lower_cols:
        return columns[lower_cols.index('repurchase_pred')]
    if 'repurchase' in lower_cols:
        return columns[lower_cols.index('repurchase')]
    return None

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Dataset Settings")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV File",
    type=["csv"],
    key="main_uploader"
)

df = load_data(uploaded_file)
purchase_column = find_purchase_label_column(df.columns.tolist()) if df is not None else None

if uploaded_file is not None and df is not None and not df.empty:
    st.sidebar.markdown("### Dataset Summary")
    st.sidebar.write(f"**File:** {uploaded_file.name}")
    st.sidebar.write(f"**Rows:** {len(df):,}")
    st.sidebar.write(f"**Users:** {df['user_id'].nunique():,}")
    st.sidebar.write(f"**Categories:** {df['product_category'].nunique():,}")
    st.sidebar.write(f"**Features:** {len(df.columns):,}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**File Status**")
    st.sidebar.success("CSV Loaded Successfully")

if uploaded_file is None:
    st.sidebar.info("Please upload a CSV file to start.")
    st.stop()

if df is None or df.empty:
    st.error("No data loaded or file is empty.")
    st.stop()

# =====================================================
# MAIN PAGE AFTER UPLOAD
# =====================================================

st.title("CUSTOMER REPURCHASE PREDICTION")
st.caption(
    "Predict whether a customer will make another purchase using machine learning analysis."
)

# =====================================================
# CUSTOMER SELECTION
# =====================================================

filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

with filter_col1:

    category_options = sorted(
        df["product_category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_category = st.selectbox(
        "Category",
        ["Select category"] + category_options,
        key="lookup_category"
    )

with filter_col2:

    if selected_category == "Select category":
        user_options = sorted(
            df["user_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        user_options = sorted(
            df[
                df["product_category"].astype(str)
                == selected_category
            ]["user_id"]
            .astype(str)
            .unique()
            .tolist()
        )

    selected_user = st.selectbox(
        "User ID",
        ["Select user"] + user_options,
        key="lookup_user"
    )

with filter_col3:

    st.markdown("<br>", unsafe_allow_html=True)

    if "customer_loaded" not in st.session_state:
        st.session_state.customer_loaded = False

    if "selected_customer_data" not in st.session_state:
        st.session_state.selected_customer_data = None

    if "run_prediction" not in st.session_state:
        st.session_state.run_prediction = False

    if st.button(
        "LOAD CUSTOMER",type="primary",
        use_container_width=True,
        key="load_customer_button"
    ):

        st.session_state.run_prediction = False

        if (
            selected_category == "Select category"
            or
            selected_user == "Select user"
        ):

            st.warning(
                "Please select a category and a User ID."
            )

            st.session_state.customer_loaded = False
            st.session_state.selected_customer_data = None

        else:

            customer_df = df[
                (df["product_category"].astype(str) == selected_category)
                &
                (df["user_id"].astype(str) == selected_user.strip())
            ]

            if customer_df.empty:

                st.warning(
                    "No matching customer found."
                )

                st.session_state.customer_loaded = False
                st.session_state.selected_customer_data = None

            else:

                st.session_state.customer_loaded = True

                st.session_state.selected_customer_data = (
                    customer_df.iloc[0].to_dict()
                )

st.divider()

# =====================================================
# NO CUSTOMER
# =====================================================

if not st.session_state.get(
    "customer_loaded",
    False
):
    st.info(
        "Select Category and User ID, then click LOAD CUSTOMER."
    )
    st.stop()

selected_customer = pd.Series(
    st.session_state.selected_customer_data
)

# # =====================================================
# # KPI OVERVIEW
# # =====================================================

# st.subheader("Customer Overview")

# k1, k2, k3, k4 = st.columns(4)

# with k1:
#     st.metric(
#         "Avg Purchase Value",
#         f"{selected_customer.get('avg_purchase_value', 0):,.0f}"
#     )

# with k2:
#     st.metric(
#         "Total Views",
#         int(selected_customer.get("total_view", 0))
#     )

# with k3:
#     st.metric(
#         "Wishlist",
#         int(selected_customer.get("total_wishlist", 0))
#     )

# with k4:
#     st.metric(
#         "Categories",
#         int(selected_customer.get("user_total_categories", 0))
#     )

# st.divider()

# =====================================================
# CUSTOMER INFORMATION
# =====================================================

customer_fields = [
    "category_commitment_score",
    "category_share",
    "engagement_depth_score",
    "user_total_categories",
    "avg_purchase_value",
    "rating_concentration",
    "total_wishlist",
    "category_breadth_score",
    "value_per_click",
    "total_view"
]

info_rows = []

for field in customer_fields:

    value = selected_customer.get(field, 0)

    if isinstance(value, float):
        value = round(value, 4)

    info_rows.append({
        "Feature": field,
        "Value": value
    })

customer_info_df = pd.DataFrame(info_rows)

left_col, right_col = st.columns([1.2, 0.8])

# =====================================================
# LEFT SIDE - TABLE
# =====================================================

with left_col:

    st.markdown("### Customer Features")

    st.dataframe(
        customer_info_df,
        use_container_width=True,
        hide_index=True,
        height=460
    )

# =====================================================
# RIGHT SIDE - DONUT CHART
# =====================================================

with right_col:

    st.markdown("### Customer Behavior Distribution")

    pie_features = [
        "category_commitment_score",
        "category_share",
        "engagement_depth_score",
        "user_total_categories",
        "rating_concentration",
        "total_wishlist",
        "category_breadth_score",
        "total_view"
    ]

    pie_labels = []
    pie_values = []

    for feature in pie_features:

        value = float(
            selected_customer.get(feature, 0)
        )

        if value > 0:

            pie_labels.append(
                feature.replace("_", " ").title()
            )

            pie_values.append(value)

    if len(pie_values) > 0:

        pie_fig = go.Figure(
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                hole=0.45,
                textinfo="percent",
                hovertemplate=
                "<b>%{label}</b><br>"
                + "Value: %{value}<br>"
                + "Percent: %{percent}"
                + "<extra></extra>"
            )
        )

        pie_fig.update_layout(
            height=460,
            margin=dict(
                t=10,
                b=10,
                l=10,
                r=10
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(
            pie_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No customer behavior data available."
        )

st.divider()

# st.markdown("---")
# st.subheader("Customer Behavior Analytics")
# analytics_items = []
# for metric in ['total_view', 'total_click', 'total_cart', 'total_wishlist', 'click_through_rate', 'cart_rate']:
#     if metric in selected_customer.index:
#         analytics_items.append({
#             'Metric': metric,
#             'Value': selected_customer.get(metric)
#         })
# if analytics_items:
#     st.table(pd.DataFrame(analytics_items))

if st.button("PREDICT", type="primary", use_container_width=True, key="predict_button"):
    st.session_state.run_prediction = True
# =================================================
# CUSTOMER ANALYSIS SECTION - PREDICTION
# =================================================

st.markdown("---")
st.header("🔮 Repurchase Prediction from Dataset")

if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

qualified_status = selected_customer.get('qualified_status', 'Unknown')
# st.write(f"**Qualified Status:** {qualified_status}")

if st.session_state.run_prediction:
    purchase_value = None
    if purchase_column is not None:
        purchase_value = selected_customer.get(purchase_column, None)

    if purchase_column is None:
        st.warning("Không tìm thấy cột repurchase_pred hoặc repurchase để xác định kết quả.")
    elif pd.isna(purchase_value):
        st.warning(f"Cột {purchase_column} có giá trị trống cho khách hàng này.")
    else:
        purchase_value = int(purchase_value)
        purchase_category = selected_customer.get('product_category', 'Unknown category')
        purchase_label = f"Có khả năng mua lại sản phẩm thuộc danh mục {purchase_category}" if purchase_value == 1 else "Không mua lại"
        st.subheader("📌 Kết quả PREDICT")
        st.success(f"Kết luận:  {purchase_label}  ")


else:
    st.info("Click PREDICT to view the repurchase result from the dataset.")
