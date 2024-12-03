import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app title
st.title("Optimized Regression Model Hyperparameter Tuning")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    dataframe = read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataframe.head())

    # Select features and target
    columns = list(dataframe.columns)
    target_column = st.selectbox("Select the target column (Y)", columns)
    feature_columns = st.multiselect("Select feature columns (X)", [col for col in columns if col != target_column])

    if target_column and feature_columns:
        X = dataframe[feature_columns]
        Y = dataframe[target_column]

        # Allow sampling of the dataset for faster processing
        use_sampling = st.sidebar.checkbox("Use Sampling for Faster Execution", value=True)
        if use_sampling:
            sample_fraction = st.sidebar.slider("Sample Fraction", 0.1, 1.0, 0.5)
            dataframe = dataframe.sample(frac=sample_fraction, random_state=42)
            X = dataframe[feature_columns]
            Y = dataframe[target_column]

        # Train-test split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        # Select models to tune
        all_models = [
            "Decision Tree Regressor", "Elastic Net", "AdaBoost Regressor",
            "K-Nearest Neighbors Regressor", "Lasso Regression", "Ridge Regression",
            "Linear Regression", "MLP Regressor", "Random Forest Regressor",
            "Support Vector Regressor (SVR)"
        ]
        selected_models = st.multiselect("Select Models for Hyperparameter Tuning", all_models, default=all_models)

        # Define models and hyperparameter grids
        models = {
            "Decision Tree Regressor": {
                "model": DecisionTreeRegressor(random_state=random_seed),
                "params": {
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }
            },
            "Elastic Net": {
                "model": ElasticNet(random_state=random_seed),
                "params": {
                    "alpha": [0.1, 1.0, 10.0],
                    "l1_ratio": [0.2, 0.5]
                }
            },
            "AdaBoost Regressor": {
                "model": AdaBoostRegressor(random_state=random_seed),
                "params": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                }
            },
            "K-Nearest Neighbors Regressor": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform", "distance"]
                }
            },
            "Lasso Regression": {
                "model": Lasso(random_state=random_seed),
                "params": {
                    "alpha": [0.1, 1.0, 10.0]
                }
            },
            "Ridge Regression": {
                "model": Ridge(random_state=random_seed),
                "params": {
                    "alpha": [0.1, 1.0, 10.0]
                }
            },
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}  # Linear Regression has no hyperparameters
            },
            "MLP Regressor": {
                "model": MLPRegressor(random_state=random_seed),
                "params": {
                    "hidden_layer_sizes": [(50,), (100,)],
                    "activation": ["relu", "tanh"],
                    "max_iter": [200]
                }
            },
            "Random Forest Regressor": {
                "model": RandomForestRegressor(random_state=random_seed),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                    "min_samples_split": [2, 5]
                }
            },
            "Support Vector Regressor (SVR)": {
                "model": SVR(),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                }
            },
        }

        # Helper function to format parameters for readability
        def format_params(params):
            """Formats the parameters dictionary into a human-readable string."""
            return ", ".join([f"{key}: {value}" for key, value in params.items()])

        # Evaluate models with hyperparameter tuning
        tuned_results = []
        for model_name in selected_models:
            st.write(f"Tuning {model_name}...")
            details = models[model_name]
            random_search = RandomizedSearchCV(
                estimator=details["model"],
                param_distributions=details["params"],
                n_iter=5,  # Limit to 5 random combinations for speed
                cv=2,  # Lower CV folds for faster execution
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                random_state=random_seed
            )
            random_search.fit(X_train, Y_train)
            best_model = random_search.best_estimator_
            best_mae = -random_search.best_score_
            tuned_results.append({
                "Model": model_name,
                "Best Parameters": format_params(random_search.best_params_),
                "Mean Absolute Error (MAE)": round(best_mae, 3)
            })

        # Create a DataFrame for tuned results
        tuned_results_df = pd.DataFrame(tuned_results).sort_values(by="Mean Absolute Error (MAE)", ascending=True)

        # Display tuned results as a table
        st.write("### Tuned Model Performance Comparison")
        st.dataframe(tuned_results_df)

        # Plot results as a bar chart
        st.write("### Tuned Model MAE Bar Chart")
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(tuned_results_df)))
        plt.barh(tuned_results_df["Model"], tuned_results_df["Mean Absolute Error (MAE)"], color=colors)
        plt.xlabel("Mean Absolute Error (MAE)")
        plt.ylabel("Model")
        plt.title("Tuned Model Performance Comparison")
        plt.gca().invert_yaxis()
        st.pyplot(plt)

        # Plot results as a line graph
        st.write("### Tuned Model MAE Line Chart")
        truncated_names = [name if len(name) <= 15 else name[:12] + "..." for name in tuned_results_df["Model"]]
        plt.figure(figsize=(10, 6))
        plt.plot(truncated_names, tuned_results_df["Mean Absolute Error (MAE)"], marker='o', linestyle='-', color='blue', label="MAE")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("Tuned Model Performance Comparison (Line Chart)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a dataset.")


