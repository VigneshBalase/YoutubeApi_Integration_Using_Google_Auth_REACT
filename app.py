import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Streamlit app
st.title("Classification Analysis App with Randomized Hyperparameter Search")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Function to read the dataset and handle large files
def load_large_csv(file, chunk_size=10000):
    try:
        # Try to load the file in chunks
        chunks = pd.read_csv(file, chunksize=chunk_size)
        data = pd.concat(chunks, ignore_index=True)
        return data
    except Exception as e:
        st.error(f"Error while loading the file: {e}")
        return None

if uploaded_file is not None:
    # Check the file size and notify the user if it's too large
    if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB limit
        st.warning("The uploaded file is too large. Please upload a file smaller than 10 MB.")
    else:
        # Try to read the file
        data = load_large_csv(uploaded_file)

        if data is not None:
            # Strip whitespaces from column names
            data.columns = data.columns.str.strip()

            # Display the data
            st.subheader("CSV Data")
            st.write(data)

            # Additional dataset checks
            st.subheader("Dataset Appropriateness Check")

            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                st.warning("The dataset contains missing values.")
                st.write(missing_values)
            else:
                st.success("No missing values detected.")

            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                st.warning(f"The dataset contains {duplicate_rows} duplicate rows.")
                if st.button("Remove Duplicates"):
                    data = data.drop_duplicates()
                    st.success("Duplicate rows removed.")
            else:
                st.success("No duplicate rows detected.")

            # Display data types
            st.subheader("Data Types")
            st.write(data.dtypes)

            # Display available columns
            st.subheader("Data Columns")
            st.write("Available columns:", data.columns)

            # Display data description
            st.subheader("Data Description")
            st.write(data.describe())

            # Handle categorical variables using one-hot encoding
            categorical_columns = data.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                st.warning("One-Hot Encoding applied to handle categorical variables.")
                encoder = OneHotEncoder(drop='first', sparse_output=False)
                data_encoded = pd.DataFrame(encoder.fit_transform(data[categorical_columns]))
                data_encoded.columns = encoder.get_feature_names_out(categorical_columns)

                # Concatenate encoded columns back to the original data
                data = pd.concat([data, data_encoded], axis=1)

                # Drop the original categorical columns
                data = data.drop(categorical_columns, axis=1)

            # Request user to select the target column
            target_column = st.selectbox("Select Target Column", [""] + list(data.columns), index=0)

            # Check for imbalanced target column
            if target_column and target_column != "":
                target_counts = data[target_column].value_counts()
                st.subheader("Target Class Distribution")
                st.bar_chart(target_counts)

                # Check for class imbalance
                imbalance_threshold = 0.1  # Adjust this threshold as needed
                class_ratios = target_counts / target_counts.sum()
                if class_ratios.min() < imbalance_threshold:
                    st.warning(f"The target column is imbalanced. Consider addressing this during model training.")
                else:
                    st.success("The target column has a balanced distribution.")

            # Proceed only if a valid target column is selected
            if target_column and target_column != "":
                # Split data into features and target
                X = data.drop(target_column, axis=1)
                y = data[target_column]

                # Convert all column names to strings (just in case they contain non-string types)
                X.columns = X.columns.astype(str)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Classification
                st.header("Classification Analysis")

                # Define classifiers with hyperparameter grids
                classifiers = {
                    "Random Forest": (RandomForestClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20, 30]
                    }),
                    "Gradient Boosting": (GradientBoostingClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }),
                    "AdaBoost": (AdaBoostClassifier(), {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }),
                    "SVM": (SVC(), {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf']
                    }),
                    "K-Nearest Neighbors": (KNeighborsClassifier(), {
                        'n_neighbors': [3, 5, 7]
                    }),
                    "Logistic Regression": (LogisticRegression(), {
                        'C': [0.1, 1, 10],
                        'solver': ['liblinear', 'lbfgs']
                    }),
                    "Decision Tree": (DecisionTreeClassifier(), {
                        'max_depth': [None, 10, 20, 30]
                    }),
                    "Naive Bayes": (GaussianNB(), {})
                }

                # Display accuracy for each classifier in a descending order
                st.subheader("Accuracy for Each Classifier (Descending Order)")

                # Calculate and store accuracies
                accuracies = {}
                for clf_name, (clf, param_grid) in classifiers.items():
                    if param_grid:  # Perform RandomizedSearchCV if param_grid is not empty
                        randomized_search = RandomizedSearchCV(clf, param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
                        randomized_search.fit(X_train, y_train)
                        clf = randomized_search.best_estimator_

                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracies[clf_name] = accuracy

                # Sort accuracies in descending order
                sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

                # Display top 10 accuracies
                top_10_classifiers = sorted_accuracies[:10]
                for clf_name, accuracy in top_10_classifiers:
                    st.write(f"{clf_name}: {accuracy}")

                # Display confusion matrix for top 5 classifiers
                st.subheader("Confusion Matrix for Top 5 Classifiers")
                for clf_name, _ in top_10_classifiers[:5]:
                    clf = classifiers[clf_name][0]  # Use classifier without hyperparameter tuning
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    confusion_mat = confusion_matrix(y_test, y_pred)
                    st.write(f"Confusion Matrix for {clf_name}:")
                    st.write(confusion_mat)

                # Write describe(), accuracy, and confusion matrix to a text file
                result_text = f"Data Description:\n{data.describe()}\n\n"
                result_text += "Accuracy for Each Classifier (Descending Order):\n"
                for clf_name, accuracy in top_10_classifiers:
                    result_text += f"{clf_name}: {accuracy}\n"

                result_text += "\nConfusion Matrix for Top 5 Classifiers:\n"
                for clf_name, _ in top_10_classifiers[:5]:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    confusion_mat = confusion_matrix(y_test, y_pred)
                    result_text += f"Confusion Matrix for {clf_name}:\n{confusion_mat}\n\n"

                # Button to download results as a text file
                if st.download_button("Download Results as Text", result_text, key="download_button"):
                    st.success("Results downloaded successfully!")
            else:
                st.info("Please select a target column from the dropdown to proceed.")
