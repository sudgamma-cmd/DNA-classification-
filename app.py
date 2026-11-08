import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

st.set_page_config(page_title="DNA Sequence Classifier", layout="centered", page_icon="🧬")

st.title("🧬 DNA Sequence Classifier & Report Generator")
st.write("Supervised machine learning model classifying DNA sequences into **Group A** or **Group B** with instant validation and assessment report downloads.")

# 1. Sequence Validation & Feature Extraction
VALID_BASES = set("ATGCU")

def validate_sequence(seq: str):
    """Checks whether the sequence contains only valid nucleotide letters (A, T, G, C, U)."""
    seq = seq.strip().upper()
    if not seq:
        return False, "Sequence cannot be empty."
    invalid_chars = sorted(list(set(seq) - VALID_BASES))
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(invalid_chars)}. Sequences must only contain A, T, G, C, or U."
    return True, ""

def extract_features(seq):
    seq = seq.strip().upper()
    length = len(seq)
    count_a = seq.count('A')
    # Count T and U together for RNA/DNA compatibility
    count_t = seq.count('T') + seq.count('U')
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

# 4. Interactive Prediction with Validation
st.subheader("2. Predict Group for Unknown Sequence")
query_seq = st.text_input("Enter DNA/RNA Sequence:", value="ATGCGCGATC")

if "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = False
    st.session_state["pred_class"] = "Group A"
    st.session_state["query_seq"] = "ATGCGCGATC"
    st.session_state["df_query"] = pd.DataFrame([{**extract_features("ATGCGCGATC")}])

if st.button("Classify Sequence", type="primary"):
    is_valid, err_msg = validate_sequence(query_seq)
    if not is_valid:
        st.error(f"❌ **Validation Error:** {err_msg}")
        st.session_state["prediction_made"] = False
    else:
        feats = extract_features(query_seq)
        df_query = pd.DataFrame([feats])
        pred = clf.predict(df_query[feature_cols])[0]
        
        st.session_state["prediction_made"] = True
        st.session_state["pred_class"] = pred
        st.session_state["query_seq"] = query_seq.strip().upper()
        st.session_state["df_query"] = df_query

if st.session_state.get("prediction_made"):
    st.write("**Extracted Features:**")
    st.dataframe(st.session_state["df_query"], use_container_width=True)
    st.success(f"### Predicted Class: **{st.session_state['pred_class']}**")

    # 5. Report Generation & Download Section
    st.divider()
    st.subheader("3. Download Assessment Report")
    st.write("Download the comprehensive assessment report containing feature tables, model accuracy, genomics applications, and critical analysis.")

    def generate_markdown_report(query, pred, query_feats):
        md_content = f"""# DNA Sequence Classification Assessment Report
**Assessment:** AI/ML DNA Sequence Analysis  
**Evaluated Query Sequence:** `{query}`  
**Predicted Class:** **{pred}**  

---

## 1. Feature Table
| Sequence ID | DNA Sequence | A | T | G | C | Length | GC Content (%) | Group |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **S1** | ATGCGATC | 2 | 2 | 2 | 2 | 8 | 50.00% | Group A |
| **S2** | ATGCGTAC | 2 | 2 | 2 | 2 | 8 | 50.00% | Group A |
| **S3** | GCGCGTAT | 1 | 2 | 3 | 2 | 8 | 62.50% | Group B |
| **S4** | GCGCGCGC | 0 | 0 | 4 | 4 | 8 | 100.00% | Group B |
| **Unknown** | {query} | {query_feats['A'].values[0]} | {query_feats['T'].values[0]} | {query_feats['G'].values[0]} | {query_feats['C'].values[0]} | {query_feats['Length'].values[0]} | {query_feats['GC_%'].values[0]}% | **Predicted: {pred}** |

---

## 2. Model Result, Parameters & Accuracy
- **Classifier Used:** Scikit-Learn DecisionTreeClassifier(criterion='gini', random_state=42)
- **Training Accuracy:** 100% (1.00) — Perfectly partitions the training dataset.
- **Decision Boundary:** Splitting on Adenine ($A \le 1.5$) partitions Group B, while $A > 1.5$ assigns Group A.

---

## 3. Real-World Genomics Application
**Pathogen Strain Identification & Antimicrobial Resistance (AMR):** Automated classification of bacterial and viral genome sequences directly from sequencing data to identify resistant mutations without requiring laboratory bacterial culture.

---

## 4. Limitation of AI/ML in Genomics
**Loss of Positional Syntax & Motifs:** Tabular frequency models ignore reading frames, regulatory motifs, and secondary structure folding, treating biological sequences merely as word bags.

---

## 5. Think & Discuss (Critical Assessment)
- **Most Useful Features:** Adenine (A) count and GC Content (%) were key discriminators ($GC=50\%$ vs $GC \ge 62.5\%$).
- **Prediction Validity:** While mathematically classified as {pred}, biologically the sequence is a mosaic with an internal GC-rich insert (`CGCG`), making the prediction sensitive to feature prioritization.
- **Behavior on Divergent Data:** Out-of-distribution sequences will produce arbitrary classifications since 4 data points cannot generalize biological sequence space.
"""
        return md_content

    current_q = st.session_state.get("query_seq", "ATGCGCGATC")
    current_p = st.session_state.get("pred_class", "Group A")
    current_f = st.session_state.get("df_query", pd.DataFrame([extract_features(current_q)]))

    col1, col2 = st.columns(2)

    with col1:
        md_report = generate_markdown_report(current_q, current_p, current_f)
        st.download_button(
            label="📄 Download Report (.md)",
            data=md_report,
            file_name="DNA_Classification_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        pdf_filename = "DNA_Sequence_Classification_Report.pdf"
        if os.path.exists(pdf_filename):
            with open(pdf_filename, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📑 Download Report (.pdf)",
                data=pdf_bytes,
                file_name="DNA_Sequence_Classification_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.download_button(
                label="📝 Download Report (.txt)",
                data=md_report,
                file_name="DNA_Classification_Report.txt",
                mime="text/plain",
                use_container_width=True
            )
