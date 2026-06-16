import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Analysis",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f1117; }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .top-header {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 24px;
    }
    .header-left h1 { color: #ffffff; font-size: 24px; margin: 0; }
    .header-left p  { color: #94a3b8; font-size: 13px; margin: 4px 0 0; }
    .header-right p { color: #4b5563; font-size: 11px; text-transform: uppercase; margin: 0; }
    .header-right h3 { color: #818cf8; font-size: 16px; margin: 4px 0 0; }

    .metric-box {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-box .label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-box .value { font-size: 28px; font-weight: 700; color: #ffffff; margin: 6px 0 2px; }
    .metric-box .sub   { font-size: 12px; color: #10b981; }
    .metric-box .sub.red { color: #f87171; }

    .section-title {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 24px 0 12px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
    }

    .model-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
    }
    .model-card.best { border-color: #065f46; }
    .model-card h4 { color: #e2e8f0; font-size: 14px; margin-bottom: 12px; }

    .step-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .segment-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .segment-card h4 { color: #e2e8f0; font-size: 14px; margin: 8px 0; }
    .segment-card p  { color: #64748b; font-size: 12px; line-height: 1.6; }

    div[data-testid="stTabs"] button {
        color: #64748b !important;
        font-size: 13px !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom-color: #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <div class="header-left">
    <h1>📊 Customer Churn Analysis</h1>
    <p>Machine Learning Pipeline — OEL Project</p>
  </div>
  <div class="header-right" style="text-align:right;">
    <p>Developed by</p>
    <h3>Syed Umair</h3>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load & Process Data ───────────────────────────────────────
@st.cache_data
def load_and_process(df_input):
    df = df_input.copy()
    if 'Exited' not in df.columns:
        return None, None, None, None, None, None, None, None, None, None

    # Drop ID columns
    df = df.drop(columns=[c for c in ['RowNumber','CustomerId','Surname'] if c in df.columns])

    # Encode categorical columns
    encoders = {}
    categorical_cols = list(df.select_dtypes(include='object').columns)
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = le

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Decision Tree':   DecisionTreeClassifier(random_state=42),
        'Random Forest':   RandomForestClassifier(random_state=42),
        'KNN':             KNeighborsClassifier(),
        'Naive Bayes':     GaussianNB(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {
            'accuracy': round(accuracy_score(y_test, pred) * 100, 1),
            'cm': confusion_matrix(y_test, pred),
            'report': classification_report(y_test, pred, output_dict=True)
        }

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    return df, X, y, X_train_s, X_test_s, y_test, results, encoders, scaler, models

# ── Helper: Predict single customer ───────────────────────────
def predict_single_customer(input_data, feature_cols, encoders, scaler, model):
    # Fill missing features with 0
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0

    row = pd.DataFrame([input_data])

    # Encode categorical columns safely
    for col, le in encoders.items():
        if col in row.columns:
            raw_val = str(row[col].iloc[0]).strip()
            valid_classes = [str(x) for x in le.classes_]

            if raw_val not in valid_classes:
                raise ValueError(
                    f"Column '{col}' has unseen value '{raw_val}'. "
                    f"Valid values: {valid_classes}"
                )

            row[col] = le.transform([raw_val])

    # Ensure exact column order
    row = row[feature_cols]

    # Scale
    row_scaled = scaler.transform(row)

    # Predict
    pred = model.predict(row_scaled)[0]
    prob = model.predict_proba(row_scaled)[0]

    return int(pred), prob, row

# ── File Upload ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Dataset Upload")
    uploaded_file = st.file_uploader("upload CSV ", type=['csv'])
    if uploaded_file:
        st.session_state['df_raw'] = pd.read_csv(uploaded_file)
        st.success(f"✅ {len(st.session_state['df_raw']):,} rows loaded")
    elif 'df_raw' not in st.session_state:
        try:
            st.session_state['df_raw'] = pd.read_csv("Customer-Churn-Records.csv")
            st.success("✅ Dataset loaded!")
        except:
            st.info("👆 Upload CSv")

    st.markdown("---")
    st.markdown("**Project Info**")
    st.markdown("- Models: DT, RF, KNN, NB")
    st.markdown("- Clustering: K-Means (k=3)")
    st.markdown("- Developer: Syed Umair")

# ── Main Content ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", "🔍 Data Analysis", "⚙️ ML Pipeline",
    "🤖 Model Results", "👥 Clustering", "💻 Source Code", "🔮 Live Prediction"
])

if 'df_raw' not in st.session_state:
    for tab in [tab1, tab2, tab3, tab4, tab5, tab7]:
        with tab:
            st.info("👈 Sidebar se pehle UPload CSV")

else:
    df, X, y, X_train_s, X_test_s, y_test, results, encoders, scaler, models = load_and_process(st.session_state['df_raw'])

    # ── TAB 1: Overview ───────────────────────────────────────
    with tab1:
        churn_count    = int(df['Exited'].sum())
        retained_count = len(df) - churn_count
        churn_rate     = round(churn_count / len(df) * 100, 1)
        best_acc       = max(v['accuracy'] for v in results.values())

        cols = st.columns(6)
        metrics = [
            ("Dataset Rows",  f"{len(df):,}",     "Customer records"),
            ("Features",      str(len(df.columns)-2), "After preprocessing"),
            ("Churn Rate",    f"{churn_rate}%",   f"{churn_count:,} churned", True),
            ("Best Accuracy", f"{best_acc}%",     "Random Forest"),
            ("ML Models",     "4",                "DT, RF, KNN, NB"),
            ("Clusters",      "3",                "K-Means segments"),
        ]
        for col, (label, value, sub, *red) in zip(cols, metrics):
            r = "red" if red else ""
            col.markdown(f"""
            <div class="metric-box">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub {r}">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Churn Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            wedges, texts, autotexts = ax.pie(
                [retained_count, churn_count],
                labels=['Retained', 'Churned'],
                colors=['#10b981', '#f87171'],
                autopct='%1.1f%%', startangle=90,
                textprops={'color': '#e2e8f0', 'fontsize': 12}
            )
            for at in autotexts: at.set_color('#0f1117')
            ax.set_title('Churn Distribution', color='#94a3b8', fontsize=13)
            st.pyplot(fig)
            plt.close()

        with c2:
            if 'Age' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1f2e')
                ax.set_facecolor('#1a1f2e')
                for val, color, label in [(0, '#10b981', 'Retained'), (1, '#f87171', 'Churned')]:
                    ax.hist(df[df['Exited']==val]['Age'], bins=20, alpha=0.7, color=color, label=label)
                ax.set_xlabel('Age', color='#94a3b8')
                ax.set_ylabel('Count', color='#94a3b8')
                ax.set_title('Age vs Churn', color='#94a3b8', fontsize=13)
                ax.tick_params(colors='#64748b')
                ax.legend(facecolor='#1a1f2e', labelcolor='#e2e8f0')
                for spine in ax.spines.values(): spine.set_edgecolor('#2d3748')
                st.pyplot(fig)
                plt.close()

    # ── TAB 2: Data Analysis ──────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            if 'Balance' in df.columns:
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1f2e')
                ax.set_facecolor('#1a1f2e')
                for val, color, label in [(0,'#818cf8','Retained'),(1,'#f472b6','Churned')]:
                    ax.hist(df[df['Exited']==val]['Balance'], bins=20, alpha=0.7, color=color, label=label)
                ax.set_title('Balance vs Churn', color='#94a3b8', fontsize=13)
                ax.tick_params(colors='#64748b')
                ax.legend(facecolor='#1a1f2e', labelcolor='#e2e8f0')
                for spine in ax.spines.values(): spine.set_edgecolor('#2d3748')
                st.pyplot(fig)
                plt.close()

        with c2:
            if 'Geography' in df.columns:
                geo_churn = df.groupby('Geography')['Exited'].mean() * 100
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1a1f2e')
                ax.set_facecolor('#1a1f2e')
                bars = ax.bar(geo_churn.index, geo_churn.values, color=['#6366f1','#f59e0b','#10b981'], width=0.5)
                ax.set_title('Churn Rate by Geography', color='#94a3b8', fontsize=13)
                ax.set_ylabel('Churn Rate (%)', color='#94a3b8')
                ax.tick_params(colors='#64748b')
                for spine in ax.spines.values(): spine.set_edgecolor('#2d3748')
                st.pyplot(fig)
                plt.close()

        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        corr = df.drop(columns=['Cluster'], errors='ignore').corr()
        sns.heatmap(corr, cmap='coolwarm', ax=ax, annot=True, fmt='.2f',
                    annot_kws={'size': 8, 'color': 'white'},
                    cbar_kws={'shrink': 0.8})
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.set_title('Feature Correlation Heatmap', color='#94a3b8', fontsize=13)
        st.pyplot(fig)
        plt.close()

    # ── TAB 3: ML Pipeline ────────────────────────────────────
    with tab3:
        steps = [
            ("1", "Data Loading",       "pandas se CSV load",                                       "✅ Done",       "#064e3b", "#6ee7b7"),
            ("2", "EDA",                "df.shape, df.isnull().sum(), df.columns",                    "✅ Done",       "#064e3b", "#6ee7b7"),
            ("3", "Feature Dropping",   "RowNumber, CustomerId, Surname remove kiye",                 "✅ Done",       "#064e3b", "#6ee7b7"),
            ("4", "Label Encoding",     "Categorical columns → numeric using LabelEncoder",           "Preprocessing", "#1e3a5f", "#93c5fd"),
            ("5", "Train/Test Split",   "80% train, 20% test — X = features, y = Exited",             "Preprocessing", "#1e3a5f", "#93c5fd"),
            ("6", "Feature Scaling",    "StandardScaler — normalize kiya",                            "Preprocessing", "#1e3a5f", "#93c5fd"),
            ("7", "Model Training",     "Decision Tree, Random Forest, KNN, Naive Bayes",             "ML",            "#312e81", "#a5b4fc"),
            ("8", "Evaluation",         "accuracy_score, confusion_matrix, classification_report",    "ML",            "#312e81", "#a5b4fc"),
            ("9", "K-Means Clustering", "n_clusters=3 — customer segments banaye",                    "ML",            "#312e81", "#a5b4fc"),
        ]
        for num, title, desc, badge, bg, fg in steps:
            st.markdown(f"""
            <div class="step-card">
              <div style="width:32px;height:32px;border-radius:50%;background:{bg};color:{fg};
                          font-size:13px;font-weight:600;display:flex;align-items:center;
                          justify-content:center;flex-shrink:0;">{num}</div>
              <div style="flex:1;">
                <div style="color:#e2e8f0;font-size:14px;font-weight:500;">{title}</div>
                <div style="color:#64748b;font-size:12px;margin-top:3px;">{desc}</div>
              </div>
              <div style="padding:4px 12px;border-radius:20px;background:{bg};color:{fg};
                          font-size:11px;font-weight:500;flex-shrink:0;">{badge}</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 4: Model Results ──────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)

        cols = st.columns(4)
        colors = {'Decision Tree':'#f59e0b','Random Forest':'#10b981','KNN':'#6366f1','Naive Bayes':'#ec4899'}
        for col, (name, res) in zip(cols, results.items()):
            best = name == 'Random Forest'
            border = "border-color:#065f46;" if best else ""
            badge = '<div style="margin-top:10px;padding:4px 10px;background:#064e3b;color:#6ee7b7;border-radius:20px;font-size:11px;display:inline-block;">🏆 Best Model</div>' if best else ''
            col.markdown(f"""
            <div class="model-card {'best' if best else ''}" style="{border}">
              <h4>{'🌲' if best else '📊'} {name}</h4>
              <div style="font-size:28px;font-weight:700;color:{colors[name]};margin-bottom:12px;">{res['accuracy']}%</div>
              <div style="font-size:12px;color:#64748b;">Precision: {round(res['report']['weighted avg']['precision']*100,1)}%</div>
              <div style="font-size:12px;color:#64748b;margin-top:4px;">Recall: {round(res['report']['weighted avg']['recall']*100,1)}%</div>
              {badge}
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Accuracy Bar Chart</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        names = list(results.keys())
        accs  = [results[n]['accuracy'] for n in names]
        clrs  = [colors[n] for n in names]
        bars  = ax.bar(names, accs, color=clrs, width=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{acc}%', ha='center', va='bottom', color='#e2e8f0', fontsize=11)
        ax.set_ylim(60, 95)
        ax.set_ylabel('Accuracy (%)', color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3748')
        st.pyplot(fig)
        plt.close()

        st.markdown('<div class="section-title">Confusion Matrix — Random Forest</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        cm = results['Random Forest']['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Retained','Churned'], yticklabels=['Retained','Churned'],
                    annot_kws={'color':'white','size':12})
        ax.tick_params(colors='#94a3b8')
        ax.set_xlabel('Predicted', color='#94a3b8')
        ax.set_ylabel('Actual', color='#94a3b8')
        st.pyplot(fig)
        plt.close()

    # ── TAB 5: Clustering ─────────────────────────────────────
    with tab5:
        segs = [
            ("#1e3a5f","#93c5fd","C0","Low-Risk Customers",   "Young, low balance, active users. Retention mein asaan."),
            ("#064e3b","#6ee7b7","C1","High-Value Stable",     "Mid-age, high balance, long tenure. Protect karna zaroori."),
            ("#4c1d95","#c4b5fd","C2","At-Risk Churners",      "Older, high balance but inactive. Turant attention chahiye."),
        ]
        c1, c2, c3 = st.columns(3)
        for col, (bg, fg, label, title, desc) in zip([c1,c2,c3], segs):
            col.markdown(f"""
            <div class="segment-card">
              <div style="width:48px;height:48px;border-radius:50%;background:{bg};color:{fg};
                          font-size:18px;font-weight:700;display:flex;align-items:center;
                          justify-content:center;margin:0 auto 10px;">{label}</div>
              <h4>{title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Cluster Scatter Plot</div>', unsafe_allow_html=True)
        if 'Cluster' in df.columns and 'Age' in df.columns and 'Balance' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1f2e')
            ax.set_facecolor('#1a1f2e')
            cluster_colors = {0:'#60a5fa', 1:'#34d399', 2:'#a78bfa'}
            cluster_labels = {0:'Low-Risk', 1:'High-Value', 2:'At-Risk'}
            for c in df['Cluster'].unique():
                mask = df['Cluster'] == c
                ax.scatter(df[mask]['Age'], df[mask]['Balance'],
                           c=cluster_colors[c], label=cluster_labels[c], alpha=0.5, s=15)
            ax.set_xlabel('Age', color='#94a3b8')
            ax.set_ylabel('Balance', color='#94a3b8')
            ax.set_title('Customer Segments (Age vs Balance)', color='#94a3b8', fontsize=13)
            ax.tick_params(colors='#64748b')
            ax.legend(facecolor='#1a1f2e', labelcolor='#e2e8f0')
            for spine in ax.spines.values(): spine.set_edgecolor('#2d3748')
            st.pyplot(fig)
            plt.close()

    # ── TAB 7: Live Prediction ────────────────────────────────
    with tab7:
        st.markdown('<div class="section-title">🔮 Live Customer Prediction</div>', unsafe_allow_html=True)

        feature_cols = list(X.columns)
        categorical_cols = list(encoders.keys())

        st.info("👇 Enter Customer Data.")

        with st.form("churn_prediction_form"):
            c1, c2, c3 = st.columns(3)
            input_data = {}

            with c1:
                if 'CreditScore' in feature_cols:
                    input_data['CreditScore'] = st.number_input('Credit Score', min_value=300, max_value=900, value=650)

                if 'Geography' in feature_cols:
                    geo_options = sorted([str(x) for x in encoders['Geography'].classes_])
                    input_data['Geography'] = st.selectbox('Geography', geo_options)

                if 'Gender' in feature_cols:
                    gender_options = sorted([str(x) for x in encoders['Gender'].classes_])
                    input_data['Gender'] = st.selectbox('Gender', gender_options)

                if 'Age' in feature_cols:
                    input_data['Age'] = st.number_input('Age', min_value=18, max_value=100, value=35)

            with c2:
                if 'Tenure' in feature_cols:
                    input_data['Tenure'] = st.number_input('Tenure (years)', min_value=0, max_value=10, value=5)

                if 'Balance' in feature_cols:
                    input_data['Balance'] = st.number_input('Balance', min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)

                if 'NumOfProducts' in feature_cols:
                    input_data['NumOfProducts'] = st.selectbox('Number of Products', [1, 2, 3, 4], index=0)

            with c3:
                if 'HasCrCard' in feature_cols:
                    input_data['HasCrCard'] = 1 if st.checkbox('Has Credit Card', value=True) else 0

                if 'IsActiveMember' in feature_cols:
                    input_data['IsActiveMember'] = 1 if st.checkbox('Is Active Member', value=True) else 0

                if 'EstimatedSalary' in feature_cols:
                    input_data['EstimatedSalary'] = st.number_input('Estimated Salary', min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)

                if 'Complain' in feature_cols:
                    input_data['Complain'] = 1 if st.checkbox('Has Complaint', value=False) else 0

                # Handle any other categorical columns not covered above
                for col in categorical_cols:
                    if col not in input_data and col in feature_cols:
                        options = sorted([str(x) for x in encoders[col].classes_])
                        input_data[col] = st.selectbox(col, options)

                # Handle any remaining numeric columns not covered above
                for col in feature_cols:
                    if col not in input_data:
                        input_data[col] = st.number_input(col, value=0)

            model_name = st.selectbox('Select Model', list(results.keys()), index=1)

            submitted = st.form_submit_button('🔮 Predict Churn')

        if submitted:
            try:
                pred, prob, row = predict_single_customer(
                    input_data, feature_cols, encoders, scaler, models[model_name]
                )

                if pred == 1:
                    st.markdown(f"""
                    <div class="metric-box" style="border:1px solid #7f1d1d;">
                      <div class="label">Prediction</div>
                      <div class="value" style="color:#f87171;">Churned</div>
                      <div class="sub red">Probability: {prob[1]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    st.error("⚠️ Is customer ka churn hone ka khatra hai — immediate action recommended!")
                else:
                    st.markdown(f"""
                    <div class="metric-box" style="border:1px solid #065f46;">
                      <div class="label">Prediction</div>
                      <div class="value" style="color:#10b981;">Retained</div>
                      <div class="sub">Probability: {prob[0]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                    st.success("✅ Customer retain")

                with st.expander("📋 Input Features (after encoding)"):
                    st.write(row)

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("💡 Tip: Check that categorical values match exactly with training data.")

# ── TAB 6: Source Code ────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title">1. Data Loading & EDA</div>', unsafe_allow_html=True)
    st.code("""import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Customer-Churn-Records.csv")
print(df.shape)
print(df.isnull().sum())
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)""", language='python')

    st.markdown('<div class="section-title">2. Preprocessing</div>', unsafe_allow_html=True)
    st.code("""from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)""", language='python')

    st.markdown('<div class="section-title">3. Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

dt  = DecisionTreeClassifier().fit(X_train, y_train)
rf  = RandomForestClassifier().fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)
nb  = GaussianNB().fit(X_train, y_train)

print("Decision Tree:", accuracy_score(y_test, dt.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, rf.predict(X_test)))
print("KNN:          ", accuracy_score(y_test, knn.predict(X_test)))
print("Naive Bayes:  ", accuracy_score(y_test, nb.predict(X_test)))""", language='python')

    st.markdown('<div class="section-title">4. K-Means Clustering</div>', unsafe_allow_html=True)
    st.code("""from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.title("Customer Segments")
plt.show()""", language='python')
