Mobile Phone Price Range Classification

A supervised machine learning project that predicts a mobile phone's price tier (budget, mid-range, high-end, or premium) from its hardware specifications, using a decision tree classifier tuned and validated with cross-validation.

Overview

Using a dataset of 2,000 phones described by 20 hardware specs (RAM, battery power, screen dimensions, camera resolution, connectivity features, etc.), this project builds a classifier that sorts phones into one of four price categories. The work covers the full applied ML workflow: auditing and handling missing data, engineering new features, checking for class imbalance, training and evaluating a baseline model, and systematically tuning model complexity with k-fold cross-validation.

Result: the tuned decision tree reaches 84.5% accuracy on a held-out test set, with a 5-fold cross-validation mean accuracy of 85% across the full dataset.

Dataset
2,000 phones, 20 feature columns plus a target label (price_range)
Target classes are perfectly balanced: 500 phones each in classes 0–3 (25% per class)
Most columns had a small number of missing values scattered across the dataset (no single column was missing more than a handful of records)
Approach

1. Missing data audit — Counted missing values per column before deciding on a handling strategy, since imputing blindly can bias a model or mask real relationships in the data (especially important to check before assuming data is missing at random).

2. Feature engineering — Derived two new features from existing specs:

screen_area = screen height × screen width
pixel_area = pixel height × pixel width (a proxy for display resolution)

Also cleaned up a column name (mobile_wt → weight) for readability.

3. Missing value imputation — Filled missing values in every feature column with that column's median. The target column (price_range) was deliberately excluded from imputation — since it's the label the model learns from, fabricating values for it risks training the model on invented ground truth rather than real outcomes.

4. Train/test split — Split the data 70/30 with stratification on the target, preserving the exact 25/25/25/25 class balance in both the training and test sets.

5. Baseline model — Trained a default DecisionTreeClassifier and evaluated it on the held-out test set.

6. Interpretability check — Trained a shallow (depth-3) tree and visualized it to identify the most influential features. RAM was the strongest predictor, forming the root split of the tree (ram <= 2217.5).

7. Hyperparameter tuning via cross-validation — Rather than trusting a single train/test split, used 5-fold stratified cross-validation to evaluate tree depths from 2 to 6 and compared mean accuracy across folds:

Max Depth	CV Accuracy
2	0.7495
3	0.7435
4	0.7915
5	0.8220
6	0.8395

Accuracy improves steadily as depth increases across this range, with depth 6 performing best among those tested — but this is not treated as a universal rule. The analysis explicitly reasons through why an excessively deep tree can eventually hurt generalization (it starts fitting noise and idiosyncrasies of the training data rather than the underlying pattern), and why evaluating on unseen data — rather than training accuracy alone — is the only reliable way to judge that.

Key Findings
RAM is by far the most important predictor of price tier
The model's classes were already balanced, so no resampling was needed
Cross-validation exposed a clearer, more reliable trend in the depth-vs-accuracy tradeoff than a single train/test split would have shown
Depth 6 was selected as the final model configuration, based on cross-validated performance rather than test-set accuracy alone (avoiding the trap of tuning against the same split used for final evaluation)
