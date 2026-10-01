import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Real Estate Market Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_DATASET = "data/real_estate.csv"

STANDARD_COLUMNS = [
    "City",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Furnishing_Status",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"
]

ML_FEATURES = [
    "City",
    "Property_Type",
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Furnishing_Status",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
]

ML_TARGET = "Price_INR_Lakhs"

NUMERIC_FEATURES = [
    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km"
]

CATEGORICAL_FEATURES = [
    "City",
    "Property_Type",
    "Furnishing_Status"
]


# =========================================================
# COLUMN ALIASES
# =========================================================

ALIASES = {

    "City": [
        "city",
        "location_city",
        "city_name"
    ],

    "Property_Type": [
        "property_type",
        "propertytype",
        "type",
        "property"
    ],

    "BHK": [
        "bhk",
        "bedrooms",
        "bedroom",
        "bhk_count"
    ],

    "Bathrooms": [
        "bathrooms",
        "bathroom",
        "bath",
        "no_of_bathrooms"
    ],

    "Super_Area_SqFt": [
        "super_area_sqft",
        "super_area",
        "built_up_area",
        "builtup_area",
        "area_sqft",
        "area"
    ],

    "Carpet_Area_SqFt": [
        "carpet_area_sqft",
        "carpet_area",
        "carpetarea",
        "carpet_sqft"
    ],

    "Floor_Number": [
        "floor_number",
        "floor",
        "floor_no"
    ],

    "Total_Floors": [
        "total_floors",
        "floors",
        "number_of_floors"
    ],

    "Age_of_Property": [
        "age_of_property",
        "property_age",
        "age",
        "propertyage"
    ],

    "Furnishing_Status": [
        "furnishing_status",
        "furnishing",
        "furnished_status"
    ],

    "Parking": [
        "parking",
        "parking_available"
    ],

    "Lift_Available": [
        "lift_available",
        "lift",
        "elevator",
        "elevator_available"
    ],

    "Gated_Community": [
        "gated_community",
        "gated",
        "gated_society"
    ],

    "Distance_to_Metro_km": [
        "distance_to_metro_km",
        "distance_to_metro",
        "metro_distance"
    ],

    "Distance_to_City_Center_km": [
        "distance_to_city_center_km",
        "distance_to_city_center",
        "city_center_distance"
    ],

    "Price_INR_Lakhs": [
        "price_inr_lakhs",
        "price_lakhs",
        "price",
        "property_price",
        "price_inr"
    ]
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_column_name(column):
    return (
        str(column)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def price_per_sqft(df):

    if (
        "Super_Area_SqFt" not in df.columns
        or "Price_INR_Lakhs" not in df.columns
    ):
        return np.nan

    valid = df[
        (df["Super_Area_SqFt"] > 0)
        & (df["Price_INR_Lakhs"] >= 0)
    ].copy()

    if len(valid) == 0:
        return np.nan

    rates = (
        valid["Price_INR_Lakhs"] * 100000
    ) / valid["Super_Area_SqFt"]

    return rates.mean()


# =========================================================
# DATA CLEANING
# =========================================================

def clean_dataset(df):

    df = df.copy()

    original_rows = len(df)

    # Standardize column names
    df.columns = [
        clean_column_name(col)
        for col in df.columns
    ]

    # Remove completely empty rows
    df = df.dropna(how="all")

    rows_after_empty = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    rows_after_duplicates = len(df)

    # Numeric columns
    numeric_columns = [
        "BHK",
        "Bathrooms",
        "Super_Area_SqFt",
        "Carpet_Area_SqFt",
        "Floor_Number",
        "Total_Floors",
        "Age_of_Property",
        "Parking",
        "Lift_Available",
        "Gated_Community",
        "Distance_to_Metro_km",
        "Distance_to_City_Center_km",
        "Price_INR_Lakhs"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Cleaning statistics
    df.attrs["original_rows"] = original_rows

    df.attrs["empty_rows_removed"] = (
        original_rows - rows_after_empty
    )

    df.attrs["duplicate_rows_removed"] = (
        rows_after_empty - rows_after_duplicates
    )

    df.attrs["final_rows"] = len(df)

    return df


# =========================================================
# LOAD DEFAULT DATASET
# =========================================================

@st.cache_data
def load_default_dataset():

    return pd.read_csv(
        DEFAULT_DATASET
    )


# =========================================================
# AUTO COLUMN MAPPING
# =========================================================

def auto_map_columns(df):

    normalized_columns = {
        normalize_text(col): col
        for col in df.columns
    }

    mapping = {}

    for standard_col, aliases in ALIASES.items():

        for alias in aliases:

            if alias in normalized_columns:

                mapping[standard_col] = (
                    normalized_columns[alias]
                )

                break

    return mapping


# =========================================================
# APPLY COLUMN MAPPING
# =========================================================

def apply_column_mapping(df, mapping):

    result = pd.DataFrame(
        index=df.index
    )

    for standard_col, source_col in mapping.items():

        if source_col in df.columns:

            result[standard_col] = df[
                source_col
            ]

    return result


# =========================================================
# PRICE NORMALIZATION
# =========================================================

def normalize_price(df, price_unit):

    df = df.copy()

    if "Price_INR_Lakhs" not in df.columns:

        return df

    df["Price_INR_Lakhs"] = pd.to_numeric(
        df["Price_INR_Lakhs"],
        errors="coerce"
    )

    if price_unit == "INR":

        df["Price_INR_Lakhs"] = (
            df["Price_INR_Lakhs"] / 100000
        )

    elif price_unit == "Crores":

        df["Price_INR_Lakhs"] = (
            df["Price_INR_Lakhs"] * 100
        )

    return df


# =========================================================
# CREATE PREPROCESSOR
# =========================================================

def create_preprocessor():

    return ColumnTransformer(

        transformers=[

            (
                "num",
                "passthrough",
                NUMERIC_FEATURES
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )


# =========================================================
# TRAIN AND COMPARE MODELS
# =========================================================

@st.cache_resource
def train_models(df):

    model_df = df[
        ML_FEATURES + [ML_TARGET]
    ].dropna()

    X = model_df[
        ML_FEATURES
    ]

    y = model_df[
        ML_TARGET
    ]

    # 80 / 20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # =====================================================
    # LINEAR REGRESSION
    # =====================================================

    linear_pipeline = Pipeline([

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            LinearRegression()
        )
    ])

    linear_pipeline.fit(
        X_train,
        y_train
    )

    linear_pred = linear_pipeline.predict(
        X_test
    )

    linear_mae = mean_absolute_error(
        y_test,
        linear_pred
    )

    linear_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            linear_pred
        )
    )

    linear_r2 = r2_score(
        y_test,
        linear_pred
    )

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    rf_pipeline = Pipeline([

        (
            "preprocessor",
            create_preprocessor()
        ),

        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    rf_pipeline.fit(
        X_train,
        y_train
    )

    rf_pred = rf_pipeline.predict(
        X_test
    )

    rf_mae = mean_absolute_error(
        y_test,
        rf_pred
    )

    rf_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            rf_pred
        )
    )

    rf_r2 = r2_score(
        y_test,
        rf_pred
    )

    return {

        "linear_model": linear_pipeline,

        "rf_model": rf_pipeline,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "linear_pred": linear_pred,

        "rf_pred": rf_pred,

        "linear_mae": linear_mae,

        "linear_rmse": linear_rmse,

        "linear_r2": linear_r2,

        "rf_mae": rf_mae,

        "rf_rmse": rf_rmse,

        "rf_r2": rf_r2
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📁 Dataset")

dataset_choice = st.sidebar.radio(
    "Choose Dataset",
    [
        "Default Dataset",
        "Upload Custom CSV"
    ]
)


# =========================================================
# DATASET SELECTION
# =========================================================

raw_df = None
dataset_name = ""


if dataset_choice == "Default Dataset":

    try:

        raw_df = load_default_dataset()

        dataset_name = "Default Dataset"

        st.sidebar.success(
            "✅ Default dataset loaded"
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

        st.stop()


else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        raw_df = pd.read_csv(
            uploaded_file
        )

        dataset_name = uploaded_file.name

        st.sidebar.success(
            "✅ Custom dataset loaded"
        )

    else:

        st.info(
            "Please upload a CSV dataset."
        )

        st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

raw_df = clean_dataset(
    raw_df
)


# =========================================================
# COLUMN MAPPING
# =========================================================

if dataset_choice == "Upload Custom CSV":

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🔧 Column Mapping"
    )

    detected_mapping = auto_map_columns(
        raw_df
    )

    mapping = {}

    for standard_col in STANDARD_COLUMNS:

        options = [
            "Not Mapped"
        ] + list(raw_df.columns)

        default_index = 0

        if standard_col in detected_mapping:

            detected_col = detected_mapping[
                standard_col
            ]

            if detected_col in options:

                default_index = options.index(
                    detected_col
                )

        selected = st.sidebar.selectbox(

            standard_col,

            options,

            index=default_index,

            key=f"mapping_{standard_col}"
        )

        if selected != "Not Mapped":

            mapping[
                standard_col
            ] = selected

    df = apply_column_mapping(
        raw_df,
        mapping
    )

else:

    df = raw_df.copy()


# =========================================================
# PRICE UNIT
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "💰 Price Unit"
)

price_unit = st.sidebar.selectbox(
    "Source Price Unit",
    [
        "Lakhs",
        "INR",
        "Crores"
    ],
    index=0
)

df = normalize_price(
    df,
    price_unit
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "City",
    "Property_Type",
    "BHK",
    "Super_Area_SqFt",
    "Price_INR_Lakhs"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.info(
        "Please use the column mapping section "
        "for your uploaded dataset."
    )

    st.stop()


# =========================================================
# NUMERIC CONVERSION
# =========================================================

for col in [

    "BHK",
    "Bathrooms",
    "Super_Area_SqFt",
    "Carpet_Area_SqFt",
    "Floor_Number",
    "Total_Floors",
    "Age_of_Property",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Distance_to_Metro_km",
    "Distance_to_City_Center_km",
    "Price_INR_Lakhs"

]:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# =========================================================
# SIDEBAR INFORMATION
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📊 Dataset Information"
)

st.sidebar.write(
    f"Rows: **{len(df):,}**"
)

st.sidebar.write(
    f"Columns: **{len(df.columns)}**"
)

st.sidebar.write(
    f"Dataset: **{dataset_name}**"
)


# =========================================================
# MARKET FILTERS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🔎 Market Filters"
)

filtered_df = df.copy()


# City
city_options = sorted(
    df["City"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_city = st.sidebar.selectbox(
    "City",
    ["All Cities"] + city_options
)

if selected_city != "All Cities":

    filtered_df = filtered_df[
        filtered_df["City"].astype(str)
        == selected_city
    ]


# Property Type
property_options = sorted(
    df["Property_Type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_property = st.sidebar.selectbox(
    "Property Type",
    ["All Property Types"] + property_options
)

if selected_property != "All Property Types":

    filtered_df = filtered_df[
        filtered_df["Property_Type"].astype(str)
        == selected_property
    ]


# BHK
bhk_values = sorted(
    pd.to_numeric(
        df["BHK"],
        errors="coerce"
    )
    .dropna()
    .unique()
    .tolist()
)

selected_bhk = st.sidebar.selectbox(
    "BHK",
    ["All BHK"] + bhk_values
)

if selected_bhk != "All BHK":

    filtered_df = filtered_df[
        filtered_df["BHK"]
        == selected_bhk
    ]


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🏠 Real Estate Market Analytics"
)

st.subheader(
    "Interactive Real Estate Business Analytics Dashboard"
)

st.write(
    "Analyze property prices, locations, BHK trends, "
    "property types and market characteristics."
)


# =========================================================
# KPI SECTION
# =========================================================

st.markdown("---")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

property_count = len(
    filtered_df
)

avg_price = (
    filtered_df["Price_INR_Lakhs"]
    .mean()
)

median_price = (
    filtered_df["Price_INR_Lakhs"]
    .median()
)

avg_area = (
    filtered_df["Super_Area_SqFt"]
    .mean()
)

avg_rate = price_per_sqft(
    filtered_df
)


with kpi1:

    st.metric(
        "🏘️ Properties",
        f"{property_count:,}"
    )


with kpi2:

    st.metric(
        "💰 Avg Price",
        f"₹{avg_price:,.2f} Lakh"
        if pd.notna(avg_price)
        else "N/A"
    )


with kpi3:

    st.metric(
        "📊 Median Price",
        f"₹{median_price:,.2f} Lakh"
        if pd.notna(median_price)
        else "N/A"
    )


with kpi4:

    st.metric(
        "📐 Avg Area",
        f"{avg_area:,.0f} sq.ft"
        if pd.notna(avg_area)
        else "N/A"
    )


with kpi5:

    st.metric(
        "💵 Price / Sq.Ft",
        f"₹{avg_rate:,.0f}"
        if pd.notna(avg_rate)
        else "N/A"
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown("---")

st.header(
    "📊 Dataset Overview"
)

ov1, ov2, ov3 = st.columns(3)

with ov1:

    st.metric(
        "🏘️ Properties",
        f"{len(filtered_df):,}"
    )

with ov2:

    st.metric(
        "🏙️ Cities",
        filtered_df["City"].nunique()
    )

with ov3:

    st.metric(
        "📋 Columns",
        len(filtered_df.columns)
    )


# =========================================================
# MARKET ANALYTICS
# =========================================================

st.markdown("---")

st.header(
    "📈 Market Analytics"
)


# Property Type + City
col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🏢 Property Type Distribution"
    )

    type_counts = (
        filtered_df["Property_Type"]
        .value_counts()
        .reset_index()
    )

    type_counts.columns = [
        "Property Type",
        "Count"
    ]

    fig = px.bar(
        type_counts,
        x="Property Type",
        y="Count",
        title="Property Type Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    st.subheader(
        "🏙️ City-wise Average Price"
    )

    city_price = (
        filtered_df
        .groupby("City")["Price_INR_Lakhs"]
        .mean()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    city_price.columns = [
        "City",
        "Average Price"
    ]

    fig = px.bar(
        city_price,
        x="City",
        y="Average Price",
        title="Average Property Price by City",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# BHK ANALYSIS
# =========================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🛏️ BHK Distribution"
    )

    bhk_counts = (
        filtered_df["BHK"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    bhk_counts.columns = [
        "BHK",
        "Count"
    ]

    fig = px.bar(
        bhk_counts,
        x="BHK",
        y="Count",
        title="BHK Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with col2:

    st.subheader(
        "💰 BHK-wise Average Price"
    )

    bhk_price = (
        filtered_df
        .groupby("BHK")["Price_INR_Lakhs"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        bhk_price,
        x="BHK",
        y="Price_INR_Lakhs",
        title="Average Price by BHK",
        text_auto=".2f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# AREA VS PRICE
# =========================================================

st.subheader(
    "📐 Area vs Property Price"
)

plot_df = filtered_df[
    [
        "Super_Area_SqFt",
        "Price_INR_Lakhs",
        "BHK",
        "City",
        "Property_Type"
    ]
].dropna()


if len(plot_df) > 5000:

    plot_df = plot_df.sample(
        5000,
        random_state=42
    )


if len(plot_df) > 0:

    fig = px.scatter(
        plot_df,
        x="Super_Area_SqFt",
        y="Price_INR_Lakhs",
        color="Property_Type",
        hover_data=[
            "City",
            "BHK"
        ],
        title="Property Area vs Price"
    )

    fig.update_layout(
        xaxis_title="Super Area (Sq.Ft)",
        yaxis_title="Price (₹ Lakh)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# LOCALITY ANALYSIS
# =========================================================

if "Locality_Type" in filtered_df.columns:

    st.subheader(
        "📍 Locality Type Analysis"
    )

    locality_counts = (
        filtered_df["Locality_Type"]
        .value_counts()
        .reset_index()
    )

    locality_counts.columns = [
        "Locality Type",
        "Count"
    ]

    fig = px.pie(
        locality_counts,
        names="Locality Type",
        values="Count",
        title="Property Distribution by Locality Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.markdown("---")

st.header(
    "💡 Business Insights"
)


if len(filtered_df) > 0:

    common_property = (
        filtered_df["Property_Type"]
        .value_counts()
        .idxmax()
    )

    common_property_count = (
        filtered_df["Property_Type"]
        .value_counts()
        .max()
    )

    common_bhk = (
        filtered_df["BHK"]
        .value_counts()
        .idxmax()
    )

    common_bhk_count = (
        filtered_df["BHK"]
        .value_counts()
        .max()
    )

    highest_price = (
        filtered_df["Price_INR_Lakhs"]
        .max()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"""
            🏢 **Most Common Property Type**

            ### {common_property}

            Listings: **{common_property_count:,}**
            """
        )

    with col2:

        st.info(
            f"""
            🛏️ **Most Common BHK**

            ### {int(common_bhk)} BHK

            Listings: **{common_bhk_count:,}**
            """
        )

    with col3:

        st.info(
            f"""
            💰 **Highest Listed Property Price**

            ### ₹{highest_price:,.2f} Lakh
            """
        )


# =========================================================
# CITY COMPARISON
# =========================================================

st.markdown("---")

st.header(
    "📍 Locality / City Comparison"
)

st.write(
    "Compare multiple locations using property count, "
    "average price, median price, average area, "
    "average BHK and price per sq.ft."
)


available_cities = sorted(
    df["City"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


selected_locations = st.multiselect(
    "Select locations to compare",
    available_cities,
    default=(
        available_cities[:2]
        if len(available_cities) >= 2
        else available_cities
    )
)


if selected_locations:

    comparison_rows = []

    for location in selected_locations:

        location_df = df[
            df["City"].astype(str)
            == location
        ]

        comparison_rows.append({

            "Location": location,

            "Properties": len(
                location_df
            ),

            "Avg Price (₹ Lakh)": round(
                location_df[
                    "Price_INR_Lakhs"
                ].mean(),
                2
            ),

            "Median Price (₹ Lakh)": round(
                location_df[
                    "Price_INR_Lakhs"
                ].median(),
                2
            ),

            "Avg Area (Sq.Ft)": round(
                location_df[
                    "Super_Area_SqFt"
                ].mean(),
                0
            ),

            "Avg Price / Sq.Ft": round(
                price_per_sqft(
                    location_df
                ),
                0
            ),

            "Avg BHK": round(
                location_df[
                    "BHK"
                ].mean(),
                2
            )
        })


    comparison_df = pd.DataFrame(
        comparison_rows
    )


    st.subheader(
        "📊 Location Comparison"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "📌 Comparison Metrics"
    )


    metric_columns = st.columns(
        len(selected_locations)
    )


    for i, location in enumerate(
        selected_locations
    ):

        location_df = df[
            df["City"].astype(str)
            == location
        ]

        with metric_columns[i]:

            st.markdown(
                f"### 📍 {location}"
            )

            st.metric(
                "Properties",
                f"{len(location_df):,}"
            )

            st.metric(
                "Avg Price",
                f"₹{location_df['Price_INR_Lakhs'].mean():,.2f} L"
            )

            st.metric(
                "Avg Area",
                f"{location_df['Super_Area_SqFt'].mean():,.0f} sq.ft"
            )

            st.metric(
                "Price / Sq.Ft",
                f"₹{price_per_sqft(location_df):,.0f}"
            )


    # Average Price
    st.subheader(
        "💰 Average Price Comparison"
    )

    fig = px.bar(
        comparison_df,
        x="Location",
        y="Avg Price (₹ Lakh)",
        text_auto=".2f",
        title="Average Property Price by Location"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Median Price
    st.subheader(
        "📊 Median Price Comparison"
    )

    fig = px.bar(
        comparison_df,
        x="Location",
        y="Median Price (₹ Lakh)",
        text_auto=".2f",
        title="Median Property Price by Location"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Price per sqft
    st.subheader(
        "💵 Average Price per Sq.Ft"
    )

    fig = px.bar(
        comparison_df,
        x="Location",
        y="Avg Price / Sq.Ft",
        text_auto=".0f",
        title="Average Price per Sq.Ft by Location"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Boxplot
    st.subheader(
        "📦 Price Distribution"
    )

    box_df = df[
        df["City"].isin(
            selected_locations
        )
    ].copy()

    fig = px.box(
        box_df,
        x="City",
        y="Price_INR_Lakhs",
        points=False,
        title="Property Price Distribution by Location"
    )

    fig.update_layout(
        yaxis_title="Price (₹ Lakh)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MACHINE LEARNING
# =========================================================

st.markdown("---")

st.header(
    "🤖 Machine Learning — Property Price Prediction"
)

st.write(
    "Linear Regression and Random Forest Regression "
    "are compared for property price estimation."
)


ml_missing = [
    col
    for col in ML_FEATURES + [ML_TARGET]
    if col not in df.columns
]


if ml_missing:

    st.warning(
        "Machine learning is unavailable because "
        "the following columns are missing:"
    )

    st.write(
        ml_missing
    )

else:

    try:

        results = train_models(
            df
        )

        # Models
        linear_model = results[
            "linear_model"
        ]

        rf_model = results[
            "rf_model"
        ]

        # Data
        X_train = results[
            "X_train"
        ]

        X_test = results[
            "X_test"
        ]

        y_train = results[
            "y_train"
        ]

        y_test = results[
            "y_test"
        ]

        # Predictions
        linear_pred = results[
            "linear_pred"
        ]

        rf_pred = results[
            "rf_pred"
        ]

        # Metrics
        linear_mae = results[
            "linear_mae"
        ]

        linear_rmse = results[
            "linear_rmse"
        ]

        linear_r2 = results[
            "linear_r2"
        ]

        rf_mae = results[
            "rf_mae"
        ]

        rf_rmse = results[
            "rf_rmse"
        ]

        rf_r2 = results[
            "rf_r2"
        ]


        st.success(
            "✅ Machine learning models trained successfully!"
        )


        # =================================================
        # MODEL COMPARISON TABLE
        # =================================================

        st.subheader(
            "📊 Model Comparison"
        )

        comparison_models = pd.DataFrame({

            "Model": [
                "Linear Regression",
                "Random Forest"
            ],

            "MAE (₹ Lakh)": [
                linear_mae,
                rf_mae
            ],

            "RMSE (₹ Lakh)": [
                linear_rmse,
                rf_rmse
            ],

            "R² Score": [
                linear_r2,
                rf_r2
            ]
        })


        comparison_models[
            "MAE (₹ Lakh)"
        ] = comparison_models[
            "MAE (₹ Lakh)"
        ].round(2)


        comparison_models[
            "RMSE (₹ Lakh)"
        ] = comparison_models[
            "RMSE (₹ Lakh)"
        ].round(2)


        comparison_models[
            "R² Score"
        ] = comparison_models[
            "R² Score"
        ].round(3)


        st.dataframe(
            comparison_models,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # MODEL METRICS
        # =================================================

        metric_col1, metric_col2 = st.columns(2)


        with metric_col1:

            st.markdown(
                "### 📈 Linear Regression"
            )

            st.metric(
                "MAE",
                f"₹{linear_mae:,.2f} Lakh"
            )

            st.metric(
                "RMSE",
                f"₹{linear_rmse:,.2f} Lakh"
            )

            st.metric(
                "R² Score",
                f"{linear_r2:.3f}"
            )


        with metric_col2:

            st.markdown(
                "### 🌲 Random Forest"
            )

            st.metric(
                "MAE",
                f"₹{rf_mae:,.2f} Lakh"
            )

            st.metric(
                "RMSE",
                f"₹{rf_rmse:,.2f} Lakh"
            )

            st.metric(
                "R² Score",
                f"{rf_r2:.3f}"
            )


        # =================================================
        # R2 COMPARISON
        # =================================================

        st.subheader(
            "📊 R² Score Comparison"
        )

        r2_chart = pd.DataFrame({

            "Model": [
                "Linear Regression",
                "Random Forest"
            ],

            "R² Score": [
                linear_r2,
                rf_r2
            ]
        })


        fig = px.bar(
            r2_chart,
            x="Model",
            y="R² Score",
            text_auto=".3f",
            title="Model R² Score Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # MAE COMPARISON
        # =================================================

        st.subheader(
            "📉 MAE Comparison"
        )

        mae_chart = pd.DataFrame({

            "Model": [
                "Linear Regression",
                "Random Forest"
            ],

            "MAE": [
                linear_mae,
                rf_mae
            ]
        })


        fig = px.bar(
            mae_chart,
            x="Model",
            y="MAE",
            text_auto=".2f",
            title="Mean Absolute Error Comparison"
        )

        fig.update_layout(
            yaxis_title="MAE (₹ Lakh)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # ACTUAL VS PREDICTED - RANDOM FOREST
        # =================================================

        st.subheader(
            "🎯 Random Forest — Actual vs Predicted"
        )

        prediction_df = pd.DataFrame({

            "Actual Price": y_test.values,

            "Predicted Price": rf_pred
        })


        if len(prediction_df) > 3000:

            prediction_df = prediction_df.sample(
                3000,
                random_state=42
            )


        fig = px.scatter(
            prediction_df,
            x="Actual Price",
            y="Predicted Price",
            title="Actual vs Predicted Property Prices",
            opacity=0.55,
            labels={
                "Actual Price":
                    "Actual Price (₹ Lakh)",

                "Predicted Price":
                    "Predicted Price (₹ Lakh)"
            }
        )


        min_value = min(

            prediction_df[
                "Actual Price"
            ].min(),

            prediction_df[
                "Predicted Price"
            ].min()
        )


        max_value = max(

            prediction_df[
                "Actual Price"
            ].max(),

            prediction_df[
                "Predicted Price"
            ].max()
        )


        fig.add_shape(

            type="line",

            x0=min_value,

            y0=min_value,

            x1=max_value,

            y1=max_value,

            line=dict(
                dash="dash",
                width=2
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.caption(
            "The dashed line represents perfect prediction."
        )


        # =================================================
        # FEATURE IMPORTANCE
        # =================================================

        st.subheader(
            "⭐ Random Forest Feature Importance"
        )


        preprocessor = (
            rf_model
            .named_steps[
                "preprocessor"
            ]
        )


        rf_estimator = (
            rf_model
            .named_steps[
                "model"
            ]
        )


        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        importances = (
            rf_estimator
            .feature_importances_
        )


        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importances

        })


        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )


        fig = px.bar(

            importance_df.sort_values(
                "Importance"
            ),

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top 15 Important Features"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # TRAINING / TESTING INFORMATION
        # =================================================

        st.subheader(
            "🧪 Training & Testing Information"
        )


        info1, info2, info3, info4 = st.columns(4)


        with info1:

            st.metric(
                "Total Records",
                f"{len(X_train) + len(X_test):,}"
            )


        with info2:

            st.metric(
                "Training Records",
                f"{len(X_train):,}"
            )


        with info3:

            st.metric(
                "Testing Records",
                f"{len(X_test):,}"
            )


        with info4:

            st.metric(
                "Split",
                "80 / 20"
            )


        st.caption(
            "The dataset uses an 80% training and "
            "20% testing split."
        )


        # =================================================
        # MODEL SUMMARY
        # =================================================

        st.subheader(
            "📋 Model Summary"
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.markdown(
                """
                **Algorithms**

                📈 Linear Regression

                🌲 Random Forest Regression

                **Preprocessing**

                • Numerical features passed through directly

                • Categorical features encoded using One-Hot Encoding

                • Unknown categories handled safely
                """
            )


        with summary_col2:

            st.markdown(
                f"""
                **Random Forest Performance**

                📊 R² Score: **{rf_r2:.3f}**

                📉 MAE: **₹{rf_mae:.2f} Lakh**

                📐 RMSE: **₹{rf_rmse:.2f} Lakh**

                🧪 Training Samples: **{len(X_train):,}**

                🔬 Testing Samples: **{len(X_test):,}**
                """
            )


        # =================================================
        # PREDICTION MODEL SELECTION
        # =================================================

        st.subheader(
            "🔮 Predict Property Price"
        )

        st.write(
            "Select a trained model and enter property "
            "details to generate an estimated price."
        )


        selected_model_name = st.radio(

            "Select Prediction Model",

            [
                "Random Forest",
                "Linear Regression"
            ],

            horizontal=True
        )


        if selected_model_name == "Random Forest":

            selected_model = rf_model

            selected_mae = rf_mae

            selected_r2 = rf_r2

        else:

            selected_model = linear_model

            selected_mae = linear_mae

            selected_r2 = linear_r2


        st.info(
            f"Prediction model selected: **{selected_model_name}**"
        )


        # =================================================
        # PREDICTION FORM
        # =================================================

        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )


        with prediction_col1:

            city_values = sorted(
                df["City"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            prediction_city = st.selectbox(
                "City",
                city_values
            )


        with prediction_col2:

            property_values = sorted(
                df["Property_Type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            prediction_property_type = st.selectbox(
                "Property Type",
                property_values
            )


        with prediction_col3:

            prediction_bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=20,
                value=2,
                step=1
            )


        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )


        with prediction_col1:

            prediction_bathrooms = st.number_input(
                "Bathrooms",
                min_value=1.0,
                max_value=20.0,
                value=2.0,
                step=1.0
            )


        with prediction_col2:

            prediction_super_area = st.number_input(
                "Super Area (Sq.Ft)",
                min_value=100.0,
                max_value=20000.0,
                value=1200.0,
                step=50.0
            )


        with prediction_col3:

            prediction_carpet_area = st.number_input(
                "Carpet Area (Sq.Ft)",
                min_value=50.0,
                max_value=20000.0,
                value=1000.0,
                step=50.0
            )


        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )


        with prediction_col1:

            prediction_floor = st.number_input(
                "Floor Number",
                min_value=0,
                max_value=100,
                value=2,
                step=1
            )


        with prediction_col2:

            prediction_total_floors = st.number_input(
                "Total Floors",
                min_value=1,
                max_value=150,
                value=10,
                step=1
            )


        with prediction_col3:

            prediction_age = st.number_input(
                "Property Age",
                min_value=0,
                max_value=150,
                value=5,
                step=1
            )


        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )


        with prediction_col1:

            furnishing_values = sorted(
                df["Furnishing_Status"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            prediction_furnishing = st.selectbox(
                "Furnishing Status",
                furnishing_values
            )


        with prediction_col2:

            parking_values = sorted(
                df["Parking"]
                .dropna()
                .unique()
                .tolist()
            )

            prediction_parking = st.selectbox(
                "Parking",
                parking_values
            )


        with prediction_col3:

            lift_values = sorted(
                df["Lift_Available"]
                .dropna()
                .unique()
                .tolist()
            )

            prediction_lift = st.selectbox(
                "Lift Available",
                lift_values
            )


        prediction_col1, prediction_col2 = (
            st.columns(2)
        )


        with prediction_col1:

            gated_values = sorted(
                df["Gated_Community"]
                .dropna()
                .unique()
                .tolist()
            )

            prediction_gated = st.selectbox(
                "Gated Community",
                gated_values
            )


        with prediction_col2:

            prediction_metro = st.number_input(
                "Distance to Metro (km)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5
            )


        prediction_city_center = st.number_input(
            "Distance to City Center (km)",
            min_value=0.0,
            max_value=100.0,
            value=8.0,
            step=0.5
        )


        # =================================================
        # PREDICT BUTTON
        # =================================================

        if st.button(
            "🔮 Predict Property Price",
            use_container_width=True,
            type="primary"
        ):

            input_data = pd.DataFrame({

                "City": [
                    prediction_city
                ],

                "Property_Type": [
                    prediction_property_type
                ],

                "BHK": [
                    prediction_bhk
                ],

                "Bathrooms": [
                    prediction_bathrooms
                ],

                "Super_Area_SqFt": [
                    prediction_super_area
                ],

                "Carpet_Area_SqFt": [
                    prediction_carpet_area
                ],

                "Floor_Number": [
                    prediction_floor
                ],

                "Total_Floors": [
                    prediction_total_floors
                ],

                "Age_of_Property": [
                    prediction_age
                ],

                "Furnishing_Status": [
                    prediction_furnishing
                ],

                "Parking": [
                    prediction_parking
                ],

                "Lift_Available": [
                    prediction_lift
                ],

                "Gated_Community": [
                    prediction_gated
                ],

                "Distance_to_Metro_km": [
                    prediction_metro
                ],

                "Distance_to_City_Center_km": [
                    prediction_city_center
                ]
            })


            predicted_price = (
                selected_model.predict(
                    input_data
                )[0]
            )


            predicted_crore = (
                predicted_price / 100
            )


            st.markdown("---")

            st.success(
                "🎯 Property price estimated successfully!"
            )


            result1, result2 = st.columns(2)


            with result1:

                st.metric(
                    "🏠 Estimated Property Price",
                    f"₹{predicted_price:,.2f} Lakh"
                )


            with result2:

                st.metric(
                    "💰 Estimated Price",
                    f"₹{predicted_crore:,.2f} Crore"
                )


            st.info(
                f"""
                **Prediction Model:** {selected_model_name}

                **Model R² Score:** {selected_r2:.3f}

                **Model MAE:** ₹{selected_mae:.2f} Lakh
                """
            )


            st.warning(
                "⚠️ This prediction is a machine-learning "
                "estimate based on the supplied dataset. "
                "It is intended for analytical and academic "
                "purposes and should not be treated as an "
                "official property valuation."
            )


    except Exception as e:

        st.error(
            f"Machine learning error: {e}"
        )


# =========================================================
# DATA QUALITY & EXPORT
# =========================================================

st.markdown("---")

st.header(
    "🧹 Data Quality & Export"
)


dq1, dq2, dq3, dq4 = st.columns(4)


with dq1:

    st.metric(
        "Total Records",
        f"{len(df):,}"
    )


with dq2:

    st.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )


with dq3:

    missing_values = int(
        filtered_df
        .isna()
        .sum()
        .sum()
    )

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


with dq4:

    duplicate_rows = int(
        filtered_df
        .duplicated()
        .sum()
    )

    st.metric(
        "Duplicate Rows",
        f"{duplicate_rows:,}"
    )


# =========================================================
# DATA CLEANING SUMMARY
# =========================================================

st.subheader(
    "🔍 Data Cleaning Summary"
)


original_rows = df.attrs.get(
    "original_rows",
    len(df)
)

empty_removed = df.attrs.get(
    "empty_rows_removed",
    0
)

duplicates_removed = df.attrs.get(
    "duplicate_rows_removed",
    0
)


clean1, clean2, clean3, clean4 = st.columns(4)


with clean1:

    st.metric(
        "Original Rows",
        f"{original_rows:,}"
    )


with clean2:

    st.metric(
        "Empty Rows Removed",
        f"{empty_removed:,}"
    )


with clean3:

    st.metric(
        "Duplicates Removed",
        f"{duplicates_removed:,}"
    )


with clean4:

    st.metric(
        "Final Rows",
        f"{len(df):,}"
    )


# =========================================================
# MISSING VALUES
# =========================================================

missing_df = (
    filtered_df
    .isna()
    .sum()
    .reset_index()
)


missing_df.columns = [
    "Column",
    "Missing Values"
]


missing_df = missing_df[
    missing_df["Missing Values"] > 0
].sort_values(
    "Missing Values",
    ascending=False
)


if len(missing_df) > 0:

    st.subheader(
        "⚠️ Missing Values by Column"
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "✅ No missing values found in the filtered dataset."
    )


# =========================================================
# EXPORT
# =========================================================

st.subheader(
    "📥 Export Filtered Dataset"
)


csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Filtered CSV",
    data=csv_data,
    file_name="filtered_real_estate_data.csv",
    mime="text/csv",
    use_container_width=True
)


# =========================================================
# FILTERED DATA
# =========================================================

st.markdown("---")

st.header(
    "📋 Filtered Property Data"
)


st.dataframe(
    filtered_df,
    use_container_width=True,
    height=500
)


# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown("---")

st.header(
    "ℹ️ About This Project"
)


about_col1, about_col2 = st.columns(2)


with about_col1:

    st.markdown(
        """
        ### 🏠 Real Estate Market Analytics

        This application provides an interactive
        environment for analyzing residential
        property data.

        **Core Technologies**

        - Python
        - Pandas
        - NumPy
        - Plotly
        - Scikit-learn
        - Streamlit
        """
    )


with about_col2:

    st.markdown(
        """
        ### 🤖 Machine Learning

        The system compares two regression approaches:

        - Linear Regression
        - Random Forest Regression

        Property price estimation uses:

        - Location
        - Property type
        - BHK
        - Bathrooms
        - Area
        - Floor information
        - Property age
        - Furnishing
        - Parking
        - Lift
        - Gated community
        - Metro distance
        - City-center distance
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.info(
    """
    **Disclaimer:** The property prices and machine-learning
    predictions displayed by this application are based on the
    supplied dataset. The predicted price is a model estimate
    for analytical and academic purposes and should not be
    considered an official market valuation.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Real Estate Market Analytics & Price Prediction System"
)

st.caption(
    "Python • Pandas • Plotly • Scikit-learn • Streamlit"
)

st.caption(
    "Academic Project | B.Tech Computer Science & Engineering"
)