import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="DNA Sequence Classifier", layout="centered", page_icon="🧬")

st.title("🧬 DNA Sequence Classifier")
st.write("Supervised machine learning model classifying DNA sequences into **Group A** or **Group B**.")

# 1. Feature Extraction Function
def extract_features(seq):
    seq = seq.strip().upper()
    length = len(seq)
    count_a = seq.count('A')
    count_t = seq.count('T')
    count_g = seq.count('G')
    count_c = seq.count('C')
    gc_content = ((count_g + count_c) / length) * 100 if length > 0 else 0.0
    return {
        'A': count_a,
        'T': count_t,
        'G': count_g,
        'C': count_c,
        'Length': length,
        'GC_%': round(gc_content, 2)
    }

# 2. Training Data
train_data = [
    {"Seq_ID": "S1", "Sequence": "ATGCGATC", "Group": "Group A"},
    {"Seq_ID": "S2", "Sequence": "ATGCGTAC", "Group": "Group A"},
    {"Seq_ID": "S3", "Sequence": "GCGCGTAT", "Group": "Group B"},
    {"Seq_ID": "S4", "Sequence": "GCGCGCGC", "Group": "Group B"}
]

features = [{**row, **extract_features(row['Sequence'])} for row in train_data]
df_train = pd.DataFrame(features)

st.subheader("1. Training Dataset & Features")
st.dataframe(df_train, use_container_width=True)

# 3. Model Training
feature_cols = ['A', 'T', 'G', 'C', 'Length', 'GC_%']
X_train = df_train[feature_cols]
y_train = df_train['Group']

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Interactive Prediction
st.subheader("2. Predict Group for Unknown Sequence")
query_seq = st.text_input("Enter DNA Sequence:", value="ATGCGCGATC")

if st.button("Classify Sequence", type="primary"):
    if query_seq:
        feats = extract_features(query_seq)
        df_query = pd.DataFrame([feats])
        
        st.write("**Extracted Features:**")
        st.dataframe(df_query, use_container_width=True)
        
        prediction = clf.predict(df_query[feature_cols])[0]
        st.success(f"### Predicted Class: **{prediction}**")
    else:
        st.warning("Please enter a valid sequence.")
