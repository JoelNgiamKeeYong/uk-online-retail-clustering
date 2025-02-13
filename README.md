# 🛍️ Online Retail Customer Segmentation Analysis

## 🚀 Business Scenario

Understanding customer behavior and segmentation is crucial for retail businesses to optimize their marketing strategies and improve customer retention. This project develops a comprehensive customer segmentation model using the Online Retail II dataset to identify distinct customer groups based on their purchasing patterns. This information is valuable for:

- **Marketing Teams:** Developing targeted marketing campaigns for different customer segments
- **Sales Teams:** Understanding and predicting customer purchasing behaviors
- **Business Strategy:** Making data-driven decisions about product offerings and customer engagement
- **Customer Service:** Providing personalized service based on customer segment

---

## 🧠 Business Problem

Treating all customers the same way is inefficient and can lead to poor resource allocation. A data-driven segmentation approach can help:

- **Identify valuable customer segments** for targeted marketing
- **Understand customer behavior patterns** across different groups
- **Optimize marketing budgets** by focusing on the right customers
- **Improve customer retention** through personalized strategies

---

## 🛠️ Solution Approach

This project uses the RFM (Recency, Frequency, Monetary) analysis combined with K-means clustering to segment customers. The workflow includes:

### 1️⃣ **Data Collection and Preprocessing**

- **Dataset Source:** Online Retail II dataset containing transactions from a UK-based online retail store
- **Data Cleaning:**
  - Removed invalid invoices and stock codes
  - Handled missing customer IDs
  - Filtered out negative quantities and zero prices
  - Removed outliers using IQR method
- **Feature Engineering:** Created RFM metrics
  - Recency: Days since last purchase
  - Frequency: Number of purchases
  - Monetary Value: Total amount spent

### 2️⃣ **Segmentation Analysis**

- **Feature Scaling:** Standardized RFM features using StandardScaler
- **Optimal Clusters:** Used elbow method and silhouette analysis to determine optimal number of clusters
- **K-means Clustering:** Applied K-means algorithm to create distinct customer segments
- **Segment Profiling:** Analyzed characteristics of each customer segment

### 3️⃣ **Interactive Dashboard Development**

- **Streamlit App:** Created an interactive dashboard for exploring customer segments
- **Visualizations:**
  - 3D scatter plot of customer segments
  - Segment distribution analysis
  - Detailed segment metrics and characteristics
- **Download Feature:** Ability to export segmented customer data

---

## 📊 Customer Segments

| Segment   | Description                  | Strategy                                   |
| --------- | ---------------------------- | ------------------------------------------ |
| RETAIN    | High-value regular customers | Focus on retention and premium services    |
| RE-ENGAGE | Inactive customers           | Re-activation campaigns and special offers |
| NURTURE   | New or low-value customers   | Growth-focused engagement and education    |
| REWARD    | Best customers               | VIP treatment and loyalty programs         |

---

### 🔖 Key Insights

- Identified distinct customer segments with different value propositions
- Clear correlation between purchase frequency and total spend
- Significant variation in customer recency across segments
- Opportunity for targeted marketing strategies based on segment characteristics

---

## ⚠️ Limitations

1️⃣ **Time Period:** Analysis limited to the dataset's time frame

2️⃣ **Geographic Bias:** Data primarily from UK-based customers

3️⃣ **Missing Context:** No demographic or customer satisfaction data

---

## 🔄 Key Skills Demonstrated

🔹 **Data Preprocessing and Feature Engineering**
🔹 **Customer Segmentation Analysis**
🔹 **K-means Clustering**
🔹 **RFM Analysis**
🔹 **Interactive Dashboard Development**
🔹 **Data Visualization**
🔹 **Business Intelligence**

---

## 🛠️ Technical Tools & Libraries

- **Python:** Core programming language
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Scikit-learn:** Machine learning algorithms
- **Plotly:** Interactive visualizations
- **Streamlit:** Web app framework
- **Seaborn/Matplotlib:** Statistical visualizations

---

## 🚀 Final Thoughts

This project demonstrates how data science techniques can be applied to solve real business problems in retail. The interactive dashboard provides stakeholders with valuable insights into their customer base, enabling data-driven decision-making. Future work could include incorporating additional data sources, such as customer demographics or product categories, to create even more detailed customer profiles.

---

## 🔗 Project Links

- [Live Demo](https://your-streamlit-app-url)
- [GitHub Repository](https://github.com/JoelNgiamKeeYong/uk-online-retail-clustering)
- [Dataset Source](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)
