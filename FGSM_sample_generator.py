import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier

# Load the dataset
df = pd.read_csv('./data.csv')  # Adjust path as needed
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

# Encode targets for all tasks
le_class = LabelEncoder()
le_category = LabelEncoder()
le_catname = LabelEncoder()
df['Class_encoded'] = le_class.fit_transform(df['Class'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])

df['group_id'] = df.apply(lambda row: row['unique_file_id'] if row['Class'] != 'Benign' 
                          else f"benign_{row.name}", axis=1)

# Define feature columns
feature_cols = [col for col in df.columns if col not in ['Category', 'Class', 'category', 'category_name', 
                                                         'Class_encoded', 'category_encoded', 'category_name_encoded',
                                                         'unique_file_id', 'group_id']]

# Split the dataset
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
validation_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Extract test features and targets for all tasks
X_test = test_df[feature_cols]
y_test_binary = test_df['Class_encoded']
y_test_category = test_df['category_encoded']
y_test_variant = test_df['category_name_encoded']

# Load pre-trained models
tasks = ['binary', 'category', 'variant']
classifiers = ['LogisticRegression', 'RandomForest']
trained_models = {task: {clf: joblib.load(f"{task}_{clf}_model.pkl") for clf in classifiers} for task in tasks}

# Adversarial Sample Generation
np.random.seed(42)

# Select 100 random Conti samples from test set
conti_encoded = le_catname.transform(['Conti'])[0]
conti_indices = test_df[test_df['category_name_encoded'] == conti_encoded].index
selected_indices = np.random.choice(conti_indices, 100, replace=False)
X_test_conti = X_test.loc[selected_indices]
y_test_conti_binary = y_test_binary.loc[selected_indices]
y_test_conti_category = y_test_category.loc[selected_indices]
y_test_conti_variant = y_test_variant.loc[selected_indices]  # Fixed here

# Save original samples to CSV
X_test_conti.to_csv("original_fgsm_samples.csv", index=False)

# Define features not to modify
features_not_to_modify = ['svcscan.nservices', 'pslist.avg_threads', 'handles.nthread', 'dlllist.avg_dlls_per_proc', 
                          'handles.nevent', 'handles.ndirectory', 'malfind.commitCharge']
indices_not_to_modify = [X_test.columns.get_loc(feat) for feat in features_not_to_modify]
indices_to_perturb = [i for i in range(X_test.shape[1]) if i not in indices_not_to_modify]

# Use variant LR model for FGSM
scaler = trained_models['variant']['LogisticRegression'].named_steps['scaler']
logistic_clf = trained_models['variant']['LogisticRegression'].named_steps['clf']

# Scale the Conti samples
X_test_conti_scaled = scaler.transform(X_test_conti)

# Create SklearnClassifier for ART
classifier = SklearnClassifier(model=logistic_clf)

# Define FGSM attack
fgsm = FastGradientMethod(estimator=classifier, eps=0.5)

# Generate adversarial samples
X_test_adv_scaled = fgsm.generate(X_test_conti_scaled)

# Restrict perturbation to all features except those in features_not_to_modify
perturbation = X_test_adv_scaled - X_test_conti_scaled
perturbation[:, indices_not_to_modify] = 0
X_test_adv_scaled = X_test_conti_scaled + perturbation

# Inverse transform to original space
X_test_adv_raw = scaler.inverse_transform(X_test_adv_scaled)

# Convert to DataFrame with feature names
X_test_adv_df = pd.DataFrame(X_test_adv_raw, columns=feature_cols)

# Save modified samples to CSV
X_test_adv_df.to_csv("modified_fgsm_samples.csv", index=False)

# Evaluate adversarial samples on LR and RF for all tasks
for task in tasks:
    print(f"\nEvaluating adversarial samples on {task} classification...")
    for clf_name in classifiers:
        model = trained_models[task][clf_name]
        
        if task == 'binary':
            y_test_task = y_test_conti_binary
            encoder = le_class
        elif task == 'category':
            y_test_task = y_test_conti_category
            encoder = le_category
        elif task == 'variant':
            y_test_task = y_test_conti_variant
            encoder = le_catname
        
        class_names = list(encoder.classes_)
        label_values = list(range(len(class_names)))
        
        y_pred_adv = model.predict(X_test_adv_df)
        
        print(f"{task} encoder classes: {encoder.classes_}")
        print(f"Raw true labels (first 5): {y_test_task[:5]}")
        print(f"Raw predicted labels (first 5): {y_pred_adv[:5]}")
        
        print(f"\nClassification Report for {clf_name} on {task} task (Adversarial):")
        print(classification_report(y_test_task, y_pred_adv, 
                                   labels=label_values, target_names=class_names, 
                                   digits=4, zero_division=0))
        
        evasion_count = sum(y_pred_adv != y_test_task)
        print(f"Total number of evasion samples for {clf_name} on {task} task: {evasion_count} out of {len(y_test_task)}")
        
        benign_label = [cls for cls in encoder.classes_ if 'benign' in cls.lower()][0]
        benign_encoded = encoder.transform([benign_label])[0]
        benign_misclassified = sum(y_pred_adv == benign_encoded)
        print(f"Number of Conti samples misclassified as {benign_label}: {benign_misclassified}")
        
        print(f"\nPrediction Distribution for {clf_name} on {task} task:")
        pred_counts = pd.Series(y_pred_adv).value_counts().reindex(label_values, fill_value=0)
        pred_dist = pd.DataFrame({
            'Variant': class_names,
            'Count': pred_counts.values
        })
        print(pred_dist.to_string(index=False))
