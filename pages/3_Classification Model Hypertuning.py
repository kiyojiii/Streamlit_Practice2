import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app title
st.title("Classification Model Hypertuning")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    dataframe = read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataframe.head())

  # Preprocessing to handle non-numeric columns
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_columns = dataframe.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_columns:
        st.warning(f"Non-numeric columns detected: {non_numeric_columns}. They will be encoded or excluded.")
        for col in non_numeric_columns:
            if dataframe[col].nunique() < 20:  # Encode categorical columns with fewer unique values
                dataframe[col] = dataframe[col].astype("category").cat.codes
            else:
                dataframe.drop(columns=[col], inplace=True)  # Drop unsuitable columns

        # Ensure target column defaults to 'LUNG_CANCER' and features default to all other columns
        if "LUNG_CANCER" in dataframe.columns:
            target_column = st.selectbox(
                "Select target column (Y):",
                dataframe.columns,
                index=dataframe.columns.get_loc("LUNG_CANCER")  # Default to 'LUNG_CANCER'
            )
            feature_columns = st.multiselect(
                "Select feature columns (X):",
                [col for col in dataframe.columns if col != target_column],
                default=[col for col in dataframe.columns if col != "LUNG_CANCER"]  # Default to all except 'LUNG_CANCER'
            )
        else:
            st.error("The dataset does not contain the 'LUNG_CANCER' column. Please upload a valid dataset.")
    if target_column and feature_columns:
        X = dataframe[feature_columns].values
        Y = dataframe[target_column].values

        # Train-test split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        # Define models and hyperparameter grids
        models = {
            "Decision Tree": {
                "model": DecisionTreeClassifier(random_state=random_seed),
                "params": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Gaussian Naive Bayes": {
                "model": GaussianNB(),
                "params": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                }
            },
            "AdaBoost": {
                "model": AdaBoostClassifier(random_state=random_seed),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                }
            },
            "K-Nearest Neighbors": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=200, random_state=random_seed),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"]
                }
            },
            "MLP Classifier": {
                "model": MLPClassifier(max_iter=200, random_state=random_seed),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                    "activation": ["relu", "tanh", "logistic"],
                    "learning_rate": ["constant", "adaptive"]
                }
            },
            "Perceptron": {
                "model": Perceptron(random_state=random_seed),
                "params": {
                    "eta0": [0.01, 0.1, 1],
                    "max_iter": [200, 500, 1000]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=random_seed),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Support Vector Machine (SVM)": {
                "model": SVC(random_state=random_seed),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf", "poly", "sigmoid"]
                }
            }
        }

        # Helper function to format parameters for readability
        def format_params(params):
            """Formats the parameters dictionary into a human-readable string."""
            return ", ".join([f"{key}: {value}" for key, value in params.items()])

        # Evaluate models with hyperparameter tuning
        tuned_results = []
        for model_name, details in models.items():
            st.write(f"Tuning {model_name}...")
            grid_search = GridSearchCV(estimator=details["model"], param_grid=details["params"], cv=3, scoring="accuracy", n_jobs=-1)
            grid_search.fit(X_train, Y_train)
            best_model = grid_search.best_estimator_
            best_accuracy = best_model.score(X_test, Y_test)
            tuned_results.append({
                "Model": model_name,
                "Best Parameters": format_params(grid_search.best_params_),  # Convert params dict to a clean string
                "Accuracy (%)": round(best_accuracy * 100, 2)  # Round accuracy to 2 decimal places
            })

        # Create a DataFrame for tuned results
        tuned_results_df = pd.DataFrame(tuned_results).sort_values(by="Accuracy (%)", ascending=False)

        # Display tuned results as a table
        st.write("### Tuned Model Performance Comparison")
        st.dataframe(tuned_results_df)

        # Plot results as a bar chart
        st.write("### Tuned Model Accuracy Bar Chart")
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(tuned_results_df)))
        plt.barh(tuned_results_df["Model"], tuned_results_df["Accuracy (%)"], color=colors)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Model")
        plt.title("Tuned Model Performance Comparison")
        plt.gca().invert_yaxis()
        st.pyplot(plt)

        # Plot results as a line graph
        st.write("### Tuned Model Accuracy Line Chart")
        truncated_names = [name if len(name) <= 15 else name[:12] + "..." for name in tuned_results_df["Model"]]
        plt.figure(figsize=(10, 6))
        plt.plot(truncated_names, tuned_results_df["Accuracy (%)"], marker='o', linestyle='-', color='blue', label="Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Accuracy (%)")
        plt.title("Tuned Model Performance Comparison (Line Chart)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a dataset.")


