import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from sklearn.model_selection import GroupShuffleSplit


#test samples - targeted FGSM - variant model - uses conti samples only

#uses Art Module for FGSM Process
#loads pretrained models for testings and FGSM
#uses variant multiclass LR classifier for FGSM process ( can be changed)
#excludes features in "features_not_to_modify" from FGSM process
#generates CSV of adversarial samples and original samples for comparisions
# tests the adversarial samples agains the pretrained models (can be changed)
# shows accuracy drop of the models



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
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
validation_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]


#change this to 'train' instead of test for training samples
# Extract test features and targets for all tasks
X_test = test_df[feature_cols]
y_test_binary = test_df['Class_encoded']
y_test_category = test_df['category_encoded']
y_test_variant = test_df['category_name_encoded']

# Load pre-trained models ( variante category or binary)
tasks = ['variant']
classifiers = ['LogisticRegression', 'RandomForest', "KNN", "DecisionTree"]
trained_models = {task: {clf: joblib.load(f"{task}_{clf}_model.pkl") for clf in classifiers} for task in tasks}

# Adversarial Sample Generation Setup
np.random.seed(42)


#if doing the randome malware samples change to  malware_indices = train_df[train_df['Class'] != 'Benign'].index a
# Select 500 random Conti samples from test set 
conti_encoded = le_catname.transform(['Conti'])[0]
conti_indices = test_df[test_df['category_name_encoded'] == conti_encoded].index
# Check if we have enough samples
if len(conti_indices) < 500:
    print(f"Warning: Only {len(conti_indices)} Conti samples available. Using all available samples.")
    selected_indices = conti_indices
else:
    selected_indices = np.random.choice(conti_indices, 500, replace=False)

# Get the selected samples
X_test_conti = X_test.loc[selected_indices]
y_test_conti_binary = y_test_binary.loc[selected_indices]
y_test_conti_category = y_test_category.loc[selected_indices]
y_test_conti_variant = y_test_variant.loc[selected_indices]

# Keep track of the original categories
original_category = test_df.loc[selected_indices, 'Category']
original_class = test_df.loc[selected_indices, 'Class']
original_category_type = test_df.loc[selected_indices, 'category']
original_category_name = test_df.loc[selected_indices, 'category_name']

# Save original samples to CSV with category information
original_samples_df = X_test_conti.copy()
original_samples_df['Category'] = original_category.values
original_samples_df['Class'] = original_class.values
original_samples_df['category'] = original_category_type.values
original_samples_df['category_name'] = original_category_name.values
original_samples_df.to_csv("original_fgsm_samples.csv", index=False)

# Define features not to modify (these will remain unchanged)
#features_not_to_modify = []
features_not_to_modify = ["svcscan.nservices", "dlllist.avg_dlls_per_proc", "svcscan.kernel_drivers", "handles.nsection", "svcscan.shared_process_services", "malfind.commitCharge"]
indices_not_to_modify = [X_test.columns.get_loc(feat) for feat in features_not_to_modify]

# Epsilon values to test
epsilon_values = [0.1]

# For comparison of different LR models
source_models = ['variant']

# Iterate over source models
# Iterate over source models
for source_model in source_models:
    # Use the variant LR model for FGSM
    scaler = trained_models['variant']['LogisticRegression'].named_steps['scaler']
    logistic_clf = trained_models['variant']['LogisticRegression'].named_steps['clf']
    
    # Scale the Conti samples
    X_test_conti_scaled = scaler.transform(X_test_conti)
    
    # Create SklearnClassifier for ART
    classifier = SklearnClassifier(model=logistic_clf)
    
    # Get the encoded value for 'Benign' in the variant task
    benign_encoded = le_catname.transform(['Benign'])[0]
    # Create target labels array, setting all samples to target 'Benign'
    y_target = np.full(len(X_test_conti_scaled), benign_encoded)
    
    # Calculate clean accuracy for each classifier before adversarial attacks
    clean_accuracies = {}
    for task in tasks:
        for clf_name in classifiers:
            model = trained_models[task][clf_name]
            y_pred_clean = model.predict(X_test_conti)
            if task == 'binary':
                y_test_task = y_test_conti_binary
            elif task == 'category':
                y_test_task = y_test_conti_category
            elif task == 'variant':
                y_test_task = y_test_conti_variant
            clean_accuracy = (y_pred_clean == y_test_task).mean()
            clean_accuracies[(task, clf_name)] = clean_accuracy
            print(f"Clean Accuracy for {clf_name} on {task} task: {clean_accuracy:.4f}")
    
    # Iterate over epsilon values
    for eps in epsilon_values:
        print(f"\nGenerating adversarial samples with {source_model} LR model, epsilon={eps}")
        
        # Define FGSM attack with current epsilon, set to targeted
        fgsm = FastGradientMethod(estimator=classifier, eps=eps, targeted=True)
        
        # Generate adversarial samples targeting 'Benign'
        X_test_adv_scaled = fgsm.generate(X_test_conti_scaled, y=y_target)
        
        # Restrict perturbation: set to 0 for features in features_not_to_modify
        perturbation = X_test_adv_scaled - X_test_conti_scaled
        perturbation[:, indices_not_to_modify] = 0  # Set perturbation to 0 for features not to modify
        X_test_adv_scaled = X_test_conti_scaled + perturbation
        
        # Inverse transform to original space
        X_test_adv_raw = scaler.inverse_transform(X_test_adv_scaled)
        
        # Convert to DataFrame with feature names
        X_test_adv_df = pd.DataFrame(X_test_adv_raw, columns=feature_cols)
        
        # Add the original category and class information back to the adversarial samples
        X_test_adv_df['Category'] = original_category.values
        X_test_adv_df['Class'] = original_class.values
        
        # Define the column order to match the original dataset
        columns_order = ['Category'] + feature_cols + ['Class']
        X_test_adv_df = X_test_adv_df[columns_order]
        
        # Save modified samples to CSV with model and epsilon in filename
        X_test_adv_df.to_csv(f"modified_fgsm_samples_{source_model}_eps_{eps}.csv", index=False)
        
        # List to store misclassification distribution results for this combination
        misclassification_results = []
        
        # Save a feature-only version for model testing
        X_test_adv_features_only = X_test_adv_df[feature_cols]
        
        # Evaluate adversarial samples on all classifiers for the variant task
        for task in tasks:
            print(f"\nEvaluating on {task} classification (Source: {source_model}, Epsilon: {eps})")
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
                
                # Use only the feature columns for prediction
                y_pred_adv = model.predict(X_test_adv_features_only)
                
                # Calculate adversarial accuracy
                adv_accuracy = (y_pred_adv == y_test_task).mean()
                clean_accuracy = clean_accuracies[(task, clf_name)]
                accuracy_drop = clean_accuracy - adv_accuracy
                print(f"Adversarial Accuracy for {clf_name}: {adv_accuracy:.4f}")
                print(f"Accuracy Drop for {clf_name}: {accuracy_drop:.4f}")
                
                # Calculate evasion count
                evasion_count = sum(y_pred_adv != y_test_task)
                print(f"Total number of evasion samples for {clf_name}: {evasion_count} out of {len(y_test_task)}")
                
                # Calculate benign misclassifications
                benign_label = [cls for cls in encoder.classes_ if 'benign' in cls.lower()][0]
                benign_encoded = encoder.transform([benign_label])[0]
                benign_misclassified = sum(y_pred_adv == benign_encoded)
                print(f"Number of Conti samples misclassified as {benign_label}: {benign_misclassified}")
                
                # Prediction distribution
                print(f"\nPrediction Distribution for {clf_name} on {task} task:")
                pred_counts = pd.Series(y_pred_adv).value_counts().reindex(label_values, fill_value=0)
                pred_dist = pd.DataFrame({
                    'Variant': class_names,
                    'Count': pred_counts.values
                })
                print(pred_dist.to_string(index=False))
                
                # Store misclassification distribution with additional metrics
                for variant, count in zip(class_names, pred_counts.values):
                    misclassification_results.append({
                        'Source_Model': source_model,
                        'Epsilon': eps,
                        'Task': task,
                        'Eval_Model': clf_name,
                        'Variant': variant,
                        'Count': count,
                        'Evasion_Count': evasion_count,
                        'Benign_Count': benign_misclassified if variant == benign_label else 0,
                        'Clean_Accuracy': clean_accuracy,
                        'Adversarial_Accuracy': adv_accuracy,
                        'Accuracy_Drop': accuracy_drop
                    })
        
        # Save misclassification distribution to CSV for this model-epsilon pair
        misclassification_df = pd.DataFrame(misclassification_results)
        csv_filename = f"misclassification_distribution_{source_model}_eps_{eps}.csv"
        misclassification_df.to_csv(csv_filename, index=False)
        print(f"\nMisclassification distribution saved to '{csv_filename}'")
