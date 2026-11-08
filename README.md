# DNA Sequence Classification Using Machine Learning

An AI/ML workflow for classifying biological DNA sequences into functional groups based on nucleotide frequency and composition metrics.

## Problem Statement
Given labelled DNA sequences from two known cohorts:
* **Group A:** `S1: ATGCGATC`, `S2: ATGCGTAC`
* **Group B:** `S3: GCGCGTAT`, `S4: GCGCGCGC`

The model extracts quantitative sequence features (A, T, G, C counts, sequence length, and GC content percentage) and trains a supervised classification model to predict the group of the unknown sequence: **`ATGCGCGATC`**.

---

## Feature Extraction Table

| Sequence ID | Sequence | A | T | G | C | Length | GC% | Group |
|---|---|---|---|---|---|---|---|---|
| **S1** | ATGCGATC | 2 | 2 | 2 | 2 | 8 | 50.00% | Group A |
| **S2** | ATGCGTAC | 2 | 2 | 2 | 2 | 8 | 50.00% | Group A |
| **S3** | GCGCGTAT | 1 | 2 | 3 | 2 | 8 | 62.50% | Group B |
| **S4** | GCGCGCGC | 0 | 0 | 4 | 4 | 8 | 100.00% | Group B |
| **Unknown** | ATGCGCGATC | 2 | 2 | 3 | 3 | 10 | 60.00% | **Predicted: Group A** |

---

## Discussion & Critical Interpretation

1. **Which features were most useful?**
   * Nucleotide counts (**A**) and **GC%** were the primary discriminators. Group A strictly preserves $A = 2$ and $GC = 50\%$, whereas Group B shows depleted adenine ($A \le 1$) and high GC-bias ($62.5\%$ to $100\%$). Sequence length provided no initial discriminative power because all training sequences had an identical length of 8.

2. **Did the model predict correctly?**
   * The decision tree classifies `ATGCGCGATC` as **Group A** based on adenine count ($A = 2$). However, biologically, the sequence possesses an internal GC insertion (`CGCG`) raising GC content to $60\%$. Because count-based feature models ignore sequence alignment, positional order, and reading frames, the prediction represents a heuristic split rather than deep biological homology.

3. **What if the sequence is very different from training data?**
   * Out-of-distribution sequences (varying lengths, poly-nucleotide repeats, ambiguous bases like `N`) will produce arbitrary classifications because 4 training instances cannot establish a robust generalized decision boundary.

4. **Genomic Application:**
   * **Antimicrobial Resistance (AMR) & Pathogen Identification:** Automated grouping of bacterial/viral gene sequences to detect antibiotic resistance mutations directly from metagenomic samples.

5. **Limitations of AI/ML in Genomics:**
   * **Loss of Syntactic Context:** Bag-of-nucleotides ignores motifs, secondary structure, and positional dependencies. Sequence-aware architectures (CNNs, Transformers, or alignment tools like BLAST) are necessary for reliable genomic interpretation.

---

## Repository Structure
```
├── dna_classifier.py               # Main standalone Python script
├── dna_classification_notebook.ipynb # Google Colab / Jupyter Notebook
├── DNA_Sequence_Classification_Report.pdf # Final assessment report
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Documentation & report breakdown
```

## How to Run
```bash
pip install -r requirements.txt
python dna_classifier.py
```
