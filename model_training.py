# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv(r'./data.csv')

# Fill missing values
df.fillna(method="ffill", inplace=True)

# Define functions for category extraction
def find_category(file_name):
    if "-" in file_name:
        return file_name.split("-")[0]
    else:
        return file_name

def find_category_name(file_name):
    if "-" in file_name:
        parts = file_name.split("-")
        return parts[1] if len(parts) > 1 else file_name
    else:
        return file_name

def extract_unique_file_id(file_name):
    return file_name.rsplit('-', 1)[0]

# Create new columns
df["category"] = df["Category"].apply(find_category)
df["category_name"] = df["Category"].apply(find_category_name)
df["unique_file_id"] = df["Category"].apply(extract_unique_file_id)

# Encode the target variables
le_class = LabelEncoder()
le_category = LabelEncoder()
le_catname = LabelEncoder()

df['Class_encoded'] = le_class.fit_transform(df['Class'])
df['category_encoded'] = le_category.fit_transform(df['category'])
df['category_name_encoded'] = le_catname.fit_transform(df['category_name'])

# Define group_id to prevent data leakage
df['group_id'] = df.apply(lambda row: row['unique_file_id'] if row['Class'] != 'Benign' 
                          else f"benign_{row.name}", axis=1)

# Split the data into train, validation, and test sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
train_idx, temp_idx = next(gss.split(df, groups=df['group_id']))
train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.857, random_state=42)
val_idx, test_idx = next(gss_temp.split(temp_df, groups=temp_df['group_id']))
validation_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

# Print dataset shapes for verification
print("Train set shape:", train_df.shape)
print("Validation set shape:", validation_df.shape)
print("Test set shape:", test_df.shape)

# Define the classification tasks
tasks = [
    {'name': 'binary', 'target_col': 'Class_encoded', 'encoder': le_class},
    {'name': 'category', 'target_col': 'category_encoded', 'encoder': le_category},
    {'name': 'variant', 'target_col': 'category_name_encoded', 'encoder': le_catname}
]

# Define classifiers and whether scaling is required
classifiers = {
    'RandomForest': (RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=4, 
                                             min_samples_leaf=2, random_state=42), False),
    'KNN': (KNeighborsClassifier(n_neighbors=7, weights='distance'), True),
    'LogisticRegression': (LogisticRegression(penalty='l2', C=0.5, solver='liblinear', 
                                              max_iter=1000, random_state=42), True),
    'DecisionTree': (DecisionTreeClassifier(max_depth=5, min_samples_split=4, 
                                            min_samples_leaf=2, random_state=42), False)
}

# Define parameter grids for hyperparameter tuning
param_grids = {
    'RandomForest': {
        'n_estimators': [50, 75],
        'max_depth': [3, 5, 7],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [2, 3]
    },
    'KNN': {
        'n_neighbors': [7, 9, 11]
    },
    'LogisticRegression': {
        'C': [0.1, 0.5, 1]
    },
    'DecisionTree': {
        'max_depth': [3, 5],
        'min_samples_split': [6, 8],
        'min_samples_leaf': [2, 3]
    }
}

# Function to extract features and target
def get_X_y(df, target_col):
    X = df.drop(columns=['Category', 'Class', 'category', 'category_name', 
                         'Class_encoded', 'category_encoded', 'category_name_encoded',
                         'unique_file_id', 'group_id'])
    y = df[target_col]
    return X, y

# Training and evaluation loop
for task in tasks:
    print(f"\nTraining models for {task['name']} classification...")
    
    # Prepare training and test data
    X_train, y_train = get_X_y(train_df, task['target_col'])
    X_test, y_test = get_X_y(test_df, task['target_col'])
    
    for clf_name, (clf_obj, scale_required) in classifiers.items():
        print(f"Training {clf_name} for {task['name']}...")
        
        # Handle models that require scaling
        if scale_required:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', clf_obj)
            ])
            grid = {f'clf__{param}': values for param, values in param_grids[clf_name].items()}
            grid_search = GridSearchCV(pipeline, grid, cv=GroupKFold(n_splits=5),
                                       scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train, groups=train_df['group_id'])
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        else:
            grid_search = GridSearchCV(clf_obj, param_grids[clf_name], cv=GroupKFold(n_splits=5),
                                       scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train, groups=train_df['group_id'])
            best_model = grid_search.best_estimator_
            print(f"Best parameters for {clf_name}: {grid_search.best_params_}")
        
        # Predict on test set
        y_pred = best_model.predict(X_test)
        
        # Inverse transform labels for readable output
        y_test_labels = task['encoder'].inverse_transform(y_test)
        y_pred_labels = task['encoder'].inverse_transform(y_pred)
        
        # Print classification report
        print(f"\nClassification Report for {clf_name} on {task['name']} task:")
        print(classification_report(y_test_labels, y_pred_labels, digits=4))
        
        # Save the trained model
        joblib.dump(best_model, f"{task['name']}_{clf_name}_model.pkl")
        print(f"Saved model to {task['name']}_{clf_name}_model.pkl")

# Optionally save the datasets
train_df.to_csv("Train_Dataset.csv", index=False)
validation_df.to_csv("Validation_Dataset.csv", index=False)
test_df.to_csv("Test_Dataset.csv", index=False)
