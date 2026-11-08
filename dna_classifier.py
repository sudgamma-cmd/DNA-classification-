"""
DNA Sequence Classification using Machine Learning
Author: Pallipati Sudhakar
Description: Classifies DNA sequences into Group A or Group B using feature extraction and decision tree / random forest classification.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

def extract_features(seq: str) -> dict:
    """Extract nucleotide counts, length, and GC content from a DNA sequence."""
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

def main():
    # Training dataset
    train_data = [
        {"Seq_ID": "S1", "Sequence": "ATGCGATC", "Group": "Group A"},
        {"Seq_ID": "S2", "Sequence": "ATGCGTAC", "Group": "Group A"},
        {"Seq_ID": "S3", "Sequence": "GCGCGTAT", "Group": "Group B"},
        {"Seq_ID": "S4", "Sequence": "GCGCGCGC", "Group": "Group B"}
    ]
    
    # Unknown sequence for prediction
    test_data = [
        {"Seq_ID": "Unknown", "Sequence": "ATGCGCGATC"}
    ]
    
    # Feature extraction
    train_features = [{**row, **extract_features(row['Sequence'])} for row in train_data]
    df_train = pd.DataFrame(train_features)
    
    test_features = [{**row, **extract_features(row['Sequence'])} for row in test_data]
    df_test = pd.DataFrame(test_features)
    
    feature_cols = ['A', 'T', 'G', 'C', 'Length', 'GC_%']
    X_train = df_train[feature_cols]
    y_train = df_train['Group']
    X_test = df_test[feature_cols]
    
    print("================== FEATURE TABLE (TRAINING) ==================")
    print(df_train[['Seq_ID', 'Sequence', 'A', 'T', 'G', 'C', 'Length', 'GC_%', 'Group']])
    
    print("\n================== UNKNOWN SEQUENCE FEATURES =================")
    print(df_test[['Seq_ID', 'Sequence', 'A', 'T', 'G', 'C', 'Length', 'GC_%']])
    
    # Train Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    pred_unknown = clf.predict(X_test)[0]
    pred_proba = clf.predict_proba(X_test)[0]
    
    print("\n================== MODEL PERFORMANCE ========================")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Predicted Class for {test_data[0]['Sequence']}: {pred_unknown}")
    print(f"Prediction Probabilities ({clf.classes_}): {pred_proba}")
    
    print("\n================== DECISION TREE RULES ======================")
    tree_rules = export_text(clf, feature_names=feature_cols)
    print(tree_rules)

if __name__ == "__main__":
    main()
