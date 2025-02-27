# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

# Load the dataset
df = pd.read_csv('./data.csv')  # Adjust path as needed

# Remove duplicates
df.drop_duplicates(inplace=True)

# Functions to extract category and unique file information
def find_category(file_name):
    return file_name.split("-")[0] if "-" in file_name else file_name

def find_category_name(file_name):
    parts = file_name.split("-")
    return parts[1] if len(parts) > 1 and "-" in file_name else file_name

def extract_unique_file_id(file_name):
    return file_name.rsplit('-', 1)[0]

# Create new columns
df["category"] = df["Category"].apply(find_category)
df["category_name"] = df["Category"].apply(find_category_name)
df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)

# Encode target and categorical columns
le_class = LabelEncoder()
df['Class_encoded'] = le_class.fit_transform(df['Class'])

# Create a grouping column for splitting
df['group_id'] = df.apply(lambda row: row['unique_file_id'] if row['Class'] != 'Benign' 
                          else f"benign_{row.name}", axis=1)

# Define features and target
features = df.drop(columns=['Category', 'Class', 'category', 'category_name', 
                            'Class_encoded', 'category_encoded', 'category_name_encoded',
                            'unique_file_id', 'group_id'], errors='ignore')
target = df['Class_encoded']

# Split the dataset: 65% train, 5% validation, 30% test
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
validation_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Function to extract features and target
def get_features_and_target(sub_df):
    X = sub_df.drop(columns=['Category', 'Class', 'category', 'category_name', 
                             'Class_encoded', 'category_encoded', 'category_name_encoded',
                             'unique_file_id', 'group_id'], errors='ignore')
    y = sub_df['Class_encoded']
    return X, y

X_train, y_train = get_features_and_target(train_df)
X_val, y_val = get_features_and_target(validation_df)
X_test, y_test = get_features_and_target(test_df)

# Define classifiers with initial settings
rf_classifier = RandomForestClassifier(
    n_estimators=50, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=7, weights='distance')
logistic_classifier = LogisticRegression(penalty='l2', C=0.5, solver='liblinear', 
                                         max_iter=1000, random_state=42)
svm_classifier = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42)
tree_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=4, 
                                         min_samples_leaf=2, random_state=42)

# Classifier dictionary
classifiers = {
    'RandomForest': (rf_classifier, False),
    'KNN': (knn_classifier, True),
    'LogisticRegression': (logistic_classifier, True),
    'SVM': (svm_classifier, True),
    'DecisionTree': (tree_classifier, False)
}

# Hyperparameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 75],
        'max_depth': [3, 5, 7],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [2, 3]
    },
    'KNN': {'n_neighbors': [7, 9, 11]},
    'LogisticRegression': {'C': [0.1, 0.5, 1]},
    'SVM': {'C': [0.1, 0.5, 1], 'kernel': ['rbf']},
    'DecisionTree': {
        'max_depth': [3, 5],
        'min_samples_split': [6, 8],
        'min_samples_leaf': [2, 3]
    }
}

# Initialize dictionary to store trained models
trained_models = {}
train_groups = train_df['group_id']

# Train classifiers with hyperparameter tuning
for clf_name, (clf_obj, scale_required) in classifiers.items():
    print(f"\nTraining {clf_name}...")
    if scale_required:
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf_obj)])
        grid = {f'clf__{param}': values for param, values in param_grids[clf_name].items()}
        grid_search = GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=5),
                                   scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train, groups=train_groups)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        grid_search = GridSearchCV(clf_obj, param_grids[clf_name], cv=GroupKFold(n_splits=5),
                                   scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train, groups=train_groups)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    
    # Store the best model
    trained_models[clf_name] = best_model
    
    # Evaluate on test set (optional baseline)
    y_test_pred = best_model.predict(X_test)
    y_test_labels = le_class.inverse_transform(y_test)
    y_test_pred_labels = le_class.inverse_transform(y_test_pred)
    print(f"\nTest Set Classification Report for {clf_name}:")
    print(classification_report(y_test_labels, y_test_pred_labels, digits=4))

# Adversarial Sample Generation
# Set random seed for reproducibility
np.random.seed(42)

# Compute median of 'svcscan.nservices' for benign samples in training set
benign_median_svcscan = X_train[y_train == 0]['svcscan.nservices'].median()
print(f"Median svcscan.nservices for benign samples: {benign_median_svcscan}")

# Select 100 random malware samples from test set
malware_indices = np.where(y_test == 1)[0]
selected_indices = np.random.choice(malware_indices, 100, replace=False)
X_test_malware = X_test.iloc[selected_indices]
y_test_malware = y_test.iloc[selected_indices]

# **Save the original 100 samples to "original_samples.csv"**
original_samples_df = test_df.iloc[selected_indices].copy()

# Modify 'svcscan.nservices' to benign median
X_test_malware_raw = X_test_malware.copy()
X_test_malware_raw['svcscan.nservices'] = benign_median_svcscan

# Get scaler and classifier from LogisticRegression's best model
logistic_best_model = trained_models['LogisticRegression']
scaler = logistic_best_model.named_steps['scaler']
logistic_clf = logistic_best_model.named_steps['clf']

# Scale the modified samples
X_test_malware_scaled = scaler.transform(X_test_malware_raw)

# Create SklearnClassifier for ART
classifier = SklearnClassifier(model=logistic_clf)

# Define FGSM attack
fgsm = FastGradientMethod(estimator=classifier, eps=0.1)

# Generate adversarial samples
X_test_adv_scaled = fgsm.generate(X_test_malware_scaled)

# Restrict perturbation to all features except 'svcscan.nservices'
svcscan_index = X_test.columns.get_loc('svcscan.nservices')
indices_not_to_perturb = [svcscan_index]
perturbation = X_test_adv_scaled - X_test_malware_scaled
perturbation[:, indices_not_to_perturb] = 0
X_test_adv_scaled = X_test_malware_scaled + perturbation

# Inverse transform to original space
X_test_adv_raw = scaler.inverse_transform(X_test_adv_scaled)

# **Save adversarial samples to "adversarial_samples.csv"**
X_test_adv_raw_df = pd.DataFrame(X_test_adv_raw, columns=X_test.columns, index=original_samples_df.index)
adversarial_samples_df = original_samples_df.copy()
adversarial_samples_df.loc[:, X_test.columns] = X_test_adv_raw_df

# Save both CSVs
original_samples_df.to_csv("original_samples.csv", index=False)
adversarial_samples_df.to_csv("adversarial_samples.csv", index=False)
print("Saved original samples to original_samples.csv")
print("Saved adversarial samples to adversarial_samples.csv")

# Evaluate adversarial samples on all trained models
for name, best_model in trained_models.items():
    y_pred_adv = best_model.predict(X_test_adv_raw)
    print(f"\nClassifier: {name} (Adversarial)")
    print(classification_report(y_test_malware, y_pred_adv, labels=[1], target_names=["Malware"], digits=4))
    # **Identify and save samples that bypass detection**
    evasion_mask = (y_pred_adv == 0) & (y_test_malware == 1)
    print(f"Number of adversarial samples evading detection: {np.sum(evasion_mask)}")
    if np.sum(evasion_mask) > 0:
        evasion_df = adversarial_samples_df[evasion_mask]
        evasion_df.to_csv(f"{name}_evasion.csv", index=False)
        print(f"Saved {np.sum(evasion_mask)} evasion samples for {name} to {name}_evasion.csv")