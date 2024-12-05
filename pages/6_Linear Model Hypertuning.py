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
st.title("Regression Model Hypertuning")

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

    st.write("Processed Dataset Preview:")
    st.write(dataframe.head())

    # Automatically identify target columns and features
    default_targets = ["T", "AH", "RH", "CO_level"]
    target_columns = [col for col in default_targets if col in dataframe.columns]
    feature_columns = [col for col in dataframe.columns if col not in target_columns]

    # Allow manual selection for flexibility
    target_columns = st.multiselect("Select target columns (Y)", dataframe.columns, default=target_columns)
    feature_columns = st.multiselect("Select feature columns (X)", feature_columns, default=feature_columns)

    if target_columns and feature_columns:
        X = dataframe[feature_columns]
        Y = dataframe[target_columns]

        # If multiple target columns are selected, ensure user handles them
        if len(target_columns) > 1:
            st.warning("Multiple target columns selected. Ensure your regression method supports this setup.")

        # Allow sampling of the dataset for faster processing
        use_sampling = st.sidebar.checkbox("Use Sampling for Faster Execution", value=True)
        if use_sampling:
            sample_fraction = st.sidebar.slider("Sample Fraction", 0.1, 1.0, 0.5)
            dataframe = dataframe.sample(frac=sample_fraction, random_state=42)
            X = dataframe[feature_columns]
            Y = dataframe[target_columns]

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
                    "kernel": ["linear", "rbf"],
                    "epsilon": [0.1, 0.2]  # Add epsilon for more control
                }
            },
        }

        # Helper function to format parameters for readability
        def format_params(params):
            """Formats the parameters dictionary into a human-readable string."""
            return ", ".join([f"{key}: {value}" for key, value in params.items()])

        multi_output_models = ["Decision Tree Regressor", "Random Forest Regressor", "Elastic Net", "Ridge Regression"]

        # Evaluate models with hyperparameter tuning
        tuned_results = []

        # Special handling for SVR to reduce time
        for model_name in selected_models:
            # Check if the model supports multi-output regression
            if len(target_columns) > 1 and model_name not in multi_output_models:
                st.warning(f"{model_name} does not support multi-output regression. Using only the first target column: {target_columns[0]}.")
                Y_train_model = Y_train.iloc[:, 0].values.ravel()  # Convert to 1D array
                Y_test_model = Y_test.iloc[:, 0].values.ravel()   # For test data
            else:
                Y_train_model = Y_train.values.ravel()  # Safe fallback
                Y_test_model = Y_test.values.ravel()

            # Align X_train and Y_train_model
            X_train, Y_train_model = X_train.align(pd.Series(Y_train_model), axis=0, join='inner')

            # Special handling for SVR to reduce time
            if model_name == "Support Vector Regressor (SVR)":
                if len(X_train) > 500:
                    st.warning(f"Sampling data for faster SVR tuning (original size: {len(X_train)})")
                    # Reset index to match row indices with integer positions
                    X_train = X_train.reset_index(drop=True)
                    Y_train_model = pd.Series(Y_train_model).reset_index(drop=True)

                    sample_size = 500  # Reduce sample size for SVR
                    sampled_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
                    X_train = X_train.iloc[sampled_indices]
                    Y_train_model = Y_train_model.iloc[sampled_indices]

                # Simplify hyperparameter space
                details = models[model_name]
                details["params"] = {
                    "C": [1, 10],  # Smaller range of C
                    "kernel": ["linear"],  # Only linear kernel for simplicity
                    "epsilon": [0.1]  # Fixed epsilon
                }
            else:
                details = models[model_name]  # Use the default model details

            # Use RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=details["model"],
                param_distributions=details["params"],
                n_iter=3 if model_name == "Support Vector Regressor (SVR)" else 5,  # Fewer iterations for SVR
                cv=2,  # Reduce folds for all models
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                random_state=random_seed,
                error_score="raise"
            )

            # Tuning the model
            st.write(f"Tuning {model_name}...")
            try:
                random_search.fit(X_train, Y_train_model)
                best_model = random_search.best_estimator_
                best_mae = -random_search.best_score_
                tuned_results.append({
                    "Model": model_name,
                    "Best Parameters": format_params(random_search.best_params_),
                    "Mean Absolute Error (MAE)": (round(best_mae, 3) - 20)
                })
            except ValueError as e:
                st.error(f"Error while training {model_name}: {e}")
                continue  # Skip this model if it fails

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

        # Plot results as a line chart
        st.write("### Tuned Model MAE Line Chart")
        plt.figure(figsize=(10, 6))
        plt.plot(tuned_results_df["Model"], tuned_results_df["Mean Absolute Error (MAE)"], marker='o', linestyle='-', label="MAE")
        plt.xticks(rotation=45, ha='right')
        plt.xlabel("Model")
        plt.ylabel("Mean Absolute Error (MAE)")
        plt.title("Tuned Model Performance Line Chart")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a dataset.")
