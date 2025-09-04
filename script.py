import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, roc_curve, auc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE



# Create a directory to save the plots
if not os.path.exists('plots'):
    os.makedirs('plots')


#------- kNN Approach --------

data = pd.read_csv('data/forestCover.csv')

#downsample data to make it less computationally intensive
data, _ = train_test_split(data, train_size=0.05, random_state=42, stratify=data['Cover_Type'])
print("Shape of data after downsampling:", data.shape)

#drop "Facet" column because of correlation with "Aspect"
data = data.drop(columns=['Facet'])

#drop "Water_Level" column because it has cardinality of 1
#data = data.drop(columns=['Water_Level'])

#drop "Observation_ID" column
data = data.drop(columns=['Observation_ID'])

#drop "Inclination" column because of only noisy values
data = data.drop(columns=['Inclination'])

#convert categorical to binary
data['Soil_Type1'] = (data['Soil_Type1'] == 'positive').astype(int)


#Impute missing values

# Replace '?' with NaN so we can work with missing values consistently
data.replace('?', np.nan, inplace=True)

# Convert columns that are object type but should be numeric
for col in data.columns:
    if data[col].dtype == 'object':
         data[col] = pd.to_numeric(data[col], errors='ignore')

if data['Cover_Type'].isnull().any():
    print("Imputing missing values in target variable 'Cover_Type' with the mode.")
    cover_type_mode = data['Cover_Type'].mode()[0]
    data['Cover_Type'].fillna(cover_type_mode, inplace=True)


missing_values = data.isnull().sum()
print("Columns with missing values (before feature imputation):")
print(missing_values[missing_values > 0])

numerical_features = data.select_dtypes(include=np.number).columns.tolist()
categorical_features = data.select_dtypes(exclude=np.number).columns.tolist()
print("Categorical features identified:", categorical_features)
numerical_features.remove('Cover_Type')

imputer_numerical = KNNImputer(n_neighbors=5)
data[numerical_features] = imputer_numerical.fit_transform(data[numerical_features])

#the only categorical feature has no missing values


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from collections import Counter

X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original training dataset shape %s" % Counter(y_train))
counts = y_train.value_counts()
majority_class = counts.idxmax()
majority_count = counts.max()
target_samples = int(majority_count * 0.2)

sampling_strategy = {i: max(count, target_samples) for i, count in counts.items() if i != majority_class}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Resampled training dataset shape %s" % Counter(y_train))

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Original training dataset shape %s" % Counter(y_train))
# Removed the SMOTE resampling. The grid search will now run on the original imbalanced data.
# Using `weights='distance'` in the grid search effectively applies Shepard's method.


from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, cohen_kappa_score

# Definieren des Parameter-Rasters
param_grid = {
    'n_neighbors': [1, 2, 3, 6, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
n_samples_50_percent = int(0.5 * len(X_train_scaled))

halving_search = HalvingGridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    verbose=3,
    n_jobs=-1,
    return_train_score=True,
    factor=2,
    random_state=42,
    max_resources=n_samples_50_percent
)
halving_search.fit(X_train_scaled, y_train)
print("Best Parameter:", halving_search.best_params_)
print("Beste F1-macro Score ):", halving_search.best_score_)

# Visualize CV results in a table
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cv_results_df = pd.DataFrame(halving_search.cv_results_)
print("\\n--- HalvingGridSearchCV Cross-Validation Results ---")
cols_of_interest = [
    'param_n_neighbors', 'param_weights', 'param_metric',
    'mean_test_score', 'std_test_score', 'rank_test_score'
]
existing_cols = [col for col in cols_of_interest if col in cv_results_df.columns]
print(cv_results_df[existing_cols].sort_values(by='rank_test_score'))
# Visualize CV results in a table

# --- Heatmap Visualization ---
for i, metric in enumerate(cv_results_df['param_metric'].unique()):
    metric_df = cv_results_df[cv_results_df['param_metric'] == metric]
    pivot_table = metric_df.pivot_table(
        values='mean_test_score',
        index='param_n_neighbors',
        columns='param_weights'
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm", cbar_kws={'label': 'F1 Macro Score'})
    plt.title(f'Heatmap of Grid Search Results (metric={metric})')
    plt.xlabel('Weights')
    plt.ylabel('Number of Neighbors (k)')
    plt.savefig(f'plots/knn_grid_search_heatmap_{i}.png')
    plt.close()


best_knn = halving_search.best_estimator_
# We must predict on the scaled test data
y_pred_best = best_knn.predict(X_test_scaled)
cols_of_interest = [
    'param_n_neighbors', 'param_weights', 'param_metric',
    'mean_test_score', 'std_test_score', 'rank_test_score'
]
existing_cols = [col for col in cols_of_interest if col in cv_results_df.columns]
print(cv_results_df[existing_cols].sort_values(by='rank_test_score'))



# We must predict on the scaled test data
y_pred_best = best_knn.predict(X_test_scaled)

print('\\n--- Final Evaluation on the test data ---')
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"F1 Score (macro): {f1_score(y_test, y_pred_best, average='macro'):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred_best):.4f}")
print('\\nClassification Report:')
print(classification_report(y_test, y_pred_best))



#------- Classification Tree Approach --------

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.impute import KNNImputer

data_tree = pd.read_csv('data/forestCover.csv')
data_tree, _ = train_test_split(data_tree, train_size=0.05, random_state=42, stratify=data_tree['Cover_Type'])

#drop "Water_Level" column because it has cardinality of 1
#data_tree = data_tree.drop(columns=['Water_Level'])

#convert categorical to binary
data_tree['Soil_Type1'] = (data_tree['Soil_Type1'] == 'positive').astype(int)

#drop "Observation_ID" column
data_tree = data_tree.drop(columns=['Observation_ID'])

#drop "Inclination" column because of only noisy values
data_tree = data_tree.drop(columns=['Inclination'])

if data['Cover_Type'].isnull().any():
    print("Imputing missing values in target variable 'Cover_Type' with the mode.")
    cover_type_mode = data['Cover_Type'].mode()[0]
    data['Cover_Type'].fillna(cover_type_mode, inplace=True)

data_tree.replace('?', np.nan, inplace=True)

for col in data_tree.columns:
    if data_tree[col].dtype == 'object':
         data_tree[col] = pd.to_numeric(data_tree[col], errors='ignore')

missing_values = data_tree.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

numerical_features = data_tree.select_dtypes(include=np.number).columns.tolist()
categorical_features = data_tree.select_dtypes(exclude=np.number).columns.tolist()
print("Categorical features identified:", categorical_features)
numerical_features.remove('Cover_Type')

imputer_numerical = KNNImputer(n_neighbors=5)
data_tree[numerical_features] = imputer_numerical.fit_transform(data_tree[numerical_features])

#we leave categorical features as is

from sklearn.model_selection import train_test_split
from collections import Counter

print (data_tree.columns.tolist())
X = data_tree.drop('Cover_Type', axis=1)
y = data_tree['Cover_Type']
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#no scaling

print("Original training dataset shape %s" % Counter(y_train_tree))
counts = y_train_tree.value_counts()
majority_class = counts.idxmax()
majority_count = counts.max()
target_samples = int(majority_count * 0.2)

sampling_strategy = {i: max(count, target_samples) for i, count in counts.items() if i != majority_class}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_res_tree, y_res_tree = smote.fit_resample(X_train_tree, y_train_tree)
print("Resampled training dataset shape %s" % Counter(y_res_tree))



# Hyperparameter tuning with HalvingGridSearchCV for Decision Tree

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score, cohen_kappa_score

# Define the parameter grid including pre- and post-pruning parameters
param_grid_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None ,10, 30],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'ccp_alpha': [0.0, 0.001, 0.01] # Post-pruning
}

# Instantiate the halving grid search
halving_search_tree = HalvingGridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_tree,
    cv=5,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1,
    factor=2,
    random_state=42
)

halving_search_tree.fit(X_res_tree, y_res_tree)

print("--- Decision Tree HalvingGridSearchCV Results ---")
print("Best Parameters:", halving_search_tree.best_params_)
print(f"Best F1-macro Score: {halving_search_tree.best_score_:.4f}")

best_tree = halving_search_tree.best_estimator_
y_pred_best_tree = best_tree.predict(X_test_tree)

print('\n--- Final Evaluation on the Test Data ---')
print(f"Accuracy: {accuracy_score(y_test_tree, y_pred_best_tree):.4f}")
print(f"F1 Score (weighted): {f1_score(y_test_tree, y_pred_best_tree, average='weighted'):.4f}")
print(f"Cohen's Kappa: {cohen_kappa_score(y_test_tree, y_pred_best_tree):.4f}")
print('\nClassification Report:')
print(classification_report(y_test_tree, y_pred_best_tree))
print(f'\nDepth of the best tree: {best_tree.get_depth()}')

import pandas as pd
cv_results_tree_df = pd.DataFrame(halving_search_tree.cv_results_)
print("\n--- HalvingGridSearchCV Cross-Validation Results Table ---")
cols_of_interest_tree = [
    'param_criterion', 'param_max_depth', 'param_min_samples_split',
    'param_min_samples_leaf', 'param_ccp_alpha',
    'mean_test_score', 'std_test_score', 'rank_test_score'
]
existing_cols_tree = [col for col in cols_of_interest_tree if col in cv_results_tree_df.columns]
print(cv_results_tree_df[existing_cols_tree].sort_values(by='rank_test_score'))



#------- Comparison --------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_knn = best_knn.predict(X_test_scaled)
y_proba_knn = best_knn.predict_proba(X_test_scaled)

knn_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_knn),
    'Precision': precision_score(y_test, y_pred_knn, average='macro'),
    'Recall': recall_score(y_test, y_pred_knn, average='macro'),
    'F1': f1_score(y_test, y_pred_knn, average='macro'),
    'AUC': roc_auc_score(y_test, y_proba_knn, multi_class='ovr'),
    'Kappa': cohen_kappa_score(y_test, y_pred_knn)
}

y_pred_tree = best_tree.predict(X_test_tree)
y_proba_tree = best_tree.predict_proba(X_test_tree)

tree_metrics = {
    'Accuracy': accuracy_score(y_test_tree, y_pred_tree),
    'Precision': precision_score(y_test_tree, y_pred_tree, average='macro'),
    'Recall': recall_score(y_test_tree, y_pred_tree, average='macro'),
    'F1': f1_score(y_test_tree, y_pred_tree, average='macro'),
    'AUC': roc_auc_score(y_test_tree, y_proba_tree, multi_class='ovr'),
    'Kappa': cohen_kappa_score(y_test_tree, y_pred_tree)
}

metrics_df = pd.DataFrame({'kNN': knn_metrics, 'Decision Tree': tree_metrics})

plt.figure(figsize=(10, 6))
sns.heatmap(metrics_df, annot=True, fmt=".4f", cmap="coolwarm")
plt.title('Model Comparison: Performance Metrics')
plt.savefig('plots/model_comparison_heatmap.png')
plt.close()


# --- ROC Curve for kNN ---
plt.figure(figsize=(10, 8))
for i, class_label in enumerate(best_knn.classes_):
    fpr, tpr, _ = roc_curve(y_test == class_label, y_proba_knn[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for kNN')
plt.legend(loc="lower right")
plt.savefig('plots/knn_roc_curve.png')
plt.close()

# --- ROC Curve for Decision Tree ---
plt.figure(figsize=(10, 8))
for i, class_label in enumerate(best_tree.classes_):
    fpr, tpr, _ = roc_curve(y_test_tree == class_label, y_proba_tree[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend(loc="lower right")
plt.savefig('plots/decision_tree_roc_curve.png')
plt.close()