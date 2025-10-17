````markdown
# Generating SVM Tree for Unsupervised Data

**Authors:**  
Dr. Naveen Nekuri, SCIS, UoH  
Amrit Majumder, SCIS, UoH  

---

## Abstract

Traditional clustering algorithms often treat all features equally, even though some contribute little or none to the clustering process—adding unnecessary computational complexity. Another challenge lies in determining the number of clusters, often treated as a hyperparameter.  

This project proposes a **clustering algorithm using Support Vector Machines (SVM) on a Decision Tree**, based on a hierarchical clustering model. The model performs **divisive clustering** using only significant features, with **SVMs controlling the branching** to achieve optimal splits.

**Keywords:** Unsupervised Learning, Clustering

---

## 1. Introduction

Support Vector Machines (SVM) [Burges, 1998; Cortes & Vapnik, 1995] and Decision Trees [Brijain et al., 2014] are both supervised learning methods. However, this project aims to adapt them to the **unsupervised clustering** context.

Existing clustering algorithms (e.g., k-means, hierarchical clustering) often rely on predefining the number of clusters — a limitation that affects generality. While algorithms such as **DBSCAN** can automatically detect clusters [Ester et al., 1996], they lack interpretability: they do not reveal how each feature contributes to cluster formation.

Our objective is to design a clustering model that:
1. Automatically identifies the number of clusters.  
2. Is **interpretable**, revealing feature contributions to clustering.

---

## 2. Related Works

### 2.1 Interpretable Hierarchical Clustering using Unsupervised Decision Tree
Basak & Krishnapuram (2005) proposed a decision-tree-based clustering method using an **inhomogeneity measure** to determine splits. Each attribute’s contribution to data distribution homogeneity was computed using Gaussian mixtures to identify valley points as split boundaries.

### 2.2 SVM-Based Binary Tree
Elaidi et al. (2018) proposed a hierarchical clustering algorithm using **SVMs in a binary tree structure**.  
Steps include:
1. Identifying two distant points as anchors.  
2. Using nearest neighbors to form two groups.  
3. Reflecting data across axes (“open window” technique).  
4. Training an SVM to form the boundary.  

The process continues recursively, forming clusters until termination.

### 2.3 Other Approaches
Various hierarchical clustering models use Gaussian Mixture Models [Li & Nehorai, 2018], unsupervised binary trees [Fraiman et al., 2013], or neural networks [Cirrincione et al., 2020]. Applications include MRI segmentation [Vupputuri et al., 2020] and web application security [Thirumaran et al., 2019].  

Decision-tree-based SVMs have also been explored in **supervised contexts** for nonlinear classification [Nie et al., 2019] and ensemble models [Ganaie et al., 2020].

---

## 3. Proposed Model

The **SVM Tree** integrates decision-tree interpretability with SVM’s optimal boundary detection to perform **unsupervised hierarchical clustering**.

### 3.1 Attribute Selection
The attribute is selected based on **entropy minimization** from Basak & Krishnapuram (2005):

\[
E(f) = \text{Entropy based on pairwise similarity and feature removal}
\]

The feature with **minimum entropy** is chosen as the most informative.

### 3.2 Dataset Splitting
Each selected attribute’s value distribution is assumed to follow a **mixture of Gaussian curves**.  
Steps:
1. Identify valley points as rough split estimates.  
2. Temporarily label data based on sides of the valley.  
3. Train a **soft-margin SVM** to find the **optimal split boundary**.  
4. Repeat for each valley to obtain branching criteria.

### 3.3 Algorithms

#### Algorithm 1: Tree Growth

```text
Algorithm grow(Node, depth)
Input: Node, depth
sets ← Node

if depth < max_depth then
    for each s ∈ sets do
        attribute, subset ← get_split(s)
        if subset ≥ 1 then
            Node(attribute) ← subset
            grow(Node(attribute), depth + 1)
        else
            Node(attribute) ← subset
        end
    end
end
````

#### Algorithm 2: Dataset Splitting

```text
Input: sample_set
Output: Node, attribute

if sample_set.attributes > 1 then
    attribute ← entropy_measure(sample_set)
    sample_set.attribute − attribute
else
    attribute ← sample_set.attribute
end

split_points ← svm_splits(attribute)
Node ← subsets of sample_set divided using split_points
```

### 3.4 Model Complexity

* Attribute selection: **O(k × n²)**
* SVM training: **O(s × n³)**
* Tree growth: **O(n log n)**

where *k* = attributes, *n* = samples, *s* = split points.

---

## 4. Results

### 4.1 Datasets Used

| Dataset               | Samples | Features | Classes |
| --------------------- | ------- | -------- | ------- |
| Iris                  | 150     | 4        | 3       |
| Wine                  | 178     | 13       | 3       |
| Breast Cancer (Wisc.) | 569     | 30       | 2       |
| Balance Scale         | 625     | 4        | 3       |
| Liver Disorder        | 345     | 5        | 2       |
| Glass Identification  | 214     | 10       | 7       |
| Haberman’s Survival   | 306     | 3        | 2       |
| Thyroid               | 215     | 5        | 3       |
| Page Blocks           | 5473    | 10       | 5       |

### 4.2 Evaluation Procedure

Although the datasets are labeled, they are used **only for evaluation**.
Each cluster’s label is assigned as the **mode** of the true labels within that cluster, allowing accuracy comparison.

### 4.3 Performance Comparison

| Dataset       | Folds | SVM Tree (Train/Test %) | UDT (Train/Test %) |
| ------------- | ----- | ----------------------- | ------------------ |
| Iris          | 5     | 74.67 / 100             | 77.33 / 96.67      |
| Iris          | 10    | 84.67 / 100             | 78.67 / 100        |
| Iris          | 20    | 90.67 / 100             | 81.33 / 100        |
| Wine          | 5     | 75.84 / 80.55           | 57.86 / 72.22      |
| Wine          | 10    | 82.58 / 100             | 72.47 / 100        |
| Wine          | 20    | 88.20 / 100             | 73.03 / 100        |
| Breast Cancer | 10    | 72.58 / 92.98           | 81.90 / 89.47      |
| Thyroid       | 20    | 88.37 / 100             | 83.25 / 100        |
| Page Blocks   | 20    | 80.09 / 93.77           | 85.43 / 90.77      |

(*Full table available in paper; results consistent across datasets.*)

### 4.4 Visualization

```
Sepal Width → Petal Width
     ├── Cluster 1
     ├── Cluster 2
     ├── ...
     └── Cluster 7
```

**Figure 1:** SVM Tree built for the Iris dataset showing interpretable attribute-based splits.

---

## 5. Conclusion

The **SVM Tree** model performs comparably or better than the **Unsupervised Decision Tree (UDT)**.
It:

* Automatically identifies cluster count.
* Provides interpretability via feature-based decision nodes.

Despite slight underperformance in low-sample folds (due to reduced training data), it effectively achieves the intended goals of **unsupervised interpretability** and **data-driven cluster discovery**.

---

## References

* Basak, J., & Krishnapuram, R. (2005). *Interpretable hierarchical clustering by constructing an unsupervised decision tree.* IEEE TKDE, 17(1), 121–132.
* Elaidi, H. et al. (2018). *An idea of a clustering algorithm using support vector machines based on binary decision tree.* IEEE ISCV.
* Li, J., & Nehorai, A. (2018). *Gaussian mixture learning via adaptive hierarchical clustering.* Signal Processing, 150, 116–121.
* Fraiman, R. et al. (2013). *Interpretable clustering using unsupervised binary trees.* Advances in Data Analysis and Classification, 7(2), 125–145.
* Cirrincione, G. et al. (2020). *The GH-EXIN neural network for hierarchical clustering.* Neural Networks, 121, 57–73.
* ... *(Full reference list included in the original paper)*

---

## Citation

If you use or reference this work, please cite as:

```
Majumder, A., & Nekuri, N. (2025). Generating SVM Tree for Unsupervised Data. SCIS, University of Hyderabad.
```

```

---

Would you like me to **include GitHub badges** (e.g., license, Python version, last updated) and a **“How to Run”** section as if this were a real repository for the algorithm? That would make it look more like a professional open-source project README.
```
