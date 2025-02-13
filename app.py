# app.py
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Customer Segmentation Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🛍️ Customer Segmentation Analysis Dashboard")
st.markdown('[View code on GitHub](https://github.com/YourGitHubUsername)', unsafe_allow_html=True)
st.markdown("""
This dashboard analyzes customer behavior using RFM (Recency, Frequency, Monetary) metrics and K-means clustering.
The analysis is based on the Online Retail II dataset, which contains transactions from a UK-based online retail store.
""")

@st.cache_data
def load_and_preprocess_data():
    # Read data
    df = pd.read_excel("online_retail_II.xlsx", sheet_name=0)
    
    # Basic cleaning
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype("str")
    df = df[df["Invoice"].str.match("^\\d{6}$") == True]
    
    df["StockCode"] = df["StockCode"].astype("str")
    mask = (
        (df["StockCode"].str.match("^\\d{5}$") == True) |
        (df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True) |
        (df["StockCode"].str.match("^PADS$") == True)
    )
    df = df[mask]
    
    # Remove missing values and zero prices
    df = df.dropna(subset=["Customer ID"])
    df = df[df["Price"] > 0]
    
    return df

@st.cache_data
def create_rfm_data(df):
    # Calculate total sales
    df["SalesLineTotal"] = df["Quantity"] * df["Price"]
    
    # Aggregate by customer
    aggregated_df = df.groupby(by="Customer ID", as_index=False).agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
    )
    
    # Calculate recency
    max_invoice_date = aggregated_df["LastInvoiceDate"].max()
    aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days
    
    return aggregated_df

@st.cache_data
def remove_outliers(df):
    # Calculate IQR for Monetary Value and Frequency
    M_Q1, M_Q3 = df["MonetaryValue"].quantile([0.25, 0.75])
    F_Q1, F_Q3 = df["Frequency"].quantile([0.25, 0.75])
    
    M_IQR = M_Q3 - M_Q1
    F_IQR = F_Q3 - F_Q1
    
    # Filter out outliers
    non_outliers_df = df[
        (df["MonetaryValue"] <= (M_Q3 + 1.5 * M_IQR)) &
        (df["MonetaryValue"] >= (M_Q1 - 1.5 * M_IQR)) &
        (df["Frequency"] <= (F_Q3 + 1.5 * F_IQR)) &
        (df["Frequency"] >= (F_Q1 - 1.5 * F_IQR))
    ]
    
    return non_outliers_df

@st.cache_data
def perform_clustering(df):
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[["MonetaryValue", "Frequency", "Recency"]])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels
    df["Cluster"] = cluster_labels
    df["ClusterLabel"] = df["Cluster"].map({
        0: "RETAIN",
        1: "RE-ENGAGE",
        2: "NURTURE",
        3: "REWARD"
    })
    
    return df

def create_3d_scatter(df):
    fig = px.scatter_3d(
        df,
        x="MonetaryValue",
        y="Frequency",
        z="Recency",
        color="ClusterLabel",
        title="3D Customer Segmentation",
        labels={
            "MonetaryValue": "Monetary Value (£)",
            "Frequency": "Purchase Frequency",
            "Recency": "Days Since Last Purchase"
        }
    )
    fig.update_layout(legend_title_text="Customer Segments")
    return fig

def create_cluster_distribution(df):
    cluster_counts = df["ClusterLabel"].value_counts()
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Customer Segment Distribution",
        labels={"x": "Segment", "y": "Number of Customers"},
        color=cluster_counts.index
    )
    return fig

def create_segment_metrics(df):
    metrics = df.groupby("ClusterLabel").agg({
        "MonetaryValue": ["mean", "min", "max", "count"],
        "Frequency": ["mean", "min", "max"],
        "Recency": ["mean", "min", "max"]
    }).round(2)
    
    metrics.columns = [
        "Avg Monetary Value", "Min Monetary Value", "Max Monetary Value", "Customer Count",
        "Avg Frequency", "Min Frequency", "Max Frequency",
        "Avg Recency", "Min Recency", "Max Recency"
    ]
    return metrics

# Sidebar
st.sidebar.header("About This Analysis")
st.sidebar.markdown("""
This dashboard presents a customer segmentation analysis using the RFM (Recency, Frequency, Monetary) model 
and K-means clustering. The analysis helps identify different customer segments based on their purchasing behavior.

**Segments:**
- 🎯 RETAIN: High-value regular customers
- 🔄 RE-ENGAGE: Inactive customers
- 🌱 NURTURE: New or low-value customers
- 🏆 REWARD: Best customers
""")

st.sidebar.header("Technical Details")
st.sidebar.markdown("""
- Data cleaning and preprocessing
- RFM analysis
- K-means clustering (k=4)
- Outlier removal using IQR method
- Interactive visualizations with Plotly
""")

# Main content
try:
    # Load and process data
    with st.spinner('Processing data...'):
        df = load_and_preprocess_data()
        rfm_df = create_rfm_data(df)
        clean_df = remove_outliers(rfm_df)
        clustered_df = perform_clustering(clean_df)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(clustered_df):,}")
    with col2:
        st.metric("Average Order Value", f"£{clustered_df['MonetaryValue'].mean():,.2f}")
    with col3:
        st.metric("Average Purchase Frequency", f"{clustered_df['Frequency'].mean():.1f}")
    with col4:
        st.metric("Average Recency (days)", f"{clustered_df['Recency'].mean():.1f}")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["3D Segmentation", "Segment Distribution", "Detailed Analysis"])

    with tab1:
        st.plotly_chart(create_3d_scatter(clustered_df), use_container_width=True)
        st.markdown("""
        This 3D scatter plot shows how customers are grouped into different segments based on their:
        - Monetary Value (total amount spent)
        - Frequency (number of purchases)
        - Recency (days since last purchase)
        """)

    with tab2:
        st.plotly_chart(create_cluster_distribution(clustered_df), use_container_width=True)
        st.markdown("""
        The bar chart shows the distribution of customers across different segments.
        This helps understand the composition of the customer base and identify which
        segments need more attention.
        """)

    with tab3:
        st.subheader("Segment Characteristics")
        metrics_df = create_segment_metrics(clustered_df)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Download button for segmented data
        csv = clustered_df.to_csv(index=False)
        st.download_button(
            label="Download Segmented Data",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )

except FileNotFoundError:
    st.error("Error: Could not find 'online_retail_II.xlsx'. Please ensure the file is in the same directory as the app.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")