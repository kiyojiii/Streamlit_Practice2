import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Streamlit app title
st.title("Regression Model Comparison")

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

    # Encoding categorical columns with fewer unique values and dropping others
    for col in non_numeric_columns:
        if dataframe[col].nunique() < 20:  # Low cardinality: suitable for encoding
            dataframe[col] = dataframe[col].astype("category").cat.codes
        else:
            dataframe.drop(columns=[col], inplace=True)  # Drop unsuitable columns

    st.write("Processed Dataset Preview:")
    st.write(dataframe.head())

    # Define and validate target columns
    target_columns = ["T", "AH", "RH", "CO_level"]
    available_targets = [col for col in target_columns if col in dataframe.columns]
    if not available_targets:
        st.warning("Expected target columns (T, AH, RH, CO_level) not found in the dataset.")
    else:
        # Allow user to select target columns
        selected_target_columns = st.multiselect(
            "Select target columns (Y):",
            available_targets,
            default=available_targets
        )

        # Allow user to select feature columns
        feature_columns = st.multiselect(
            "Select feature columns (X):",
            [col for col in dataframe.columns if col not in selected_target_columns],
            default=[col for col in dataframe.columns if col not in selected_target_columns]
        )

        if selected_target_columns and feature_columns:
            # Subsample the data for faster evaluation if necessary
            max_sample_size = st.sidebar.number_input(
                "Max Rows for Evaluation", 
                min_value=100, 
                max_value=len(dataframe), 
                value=min(1000, len(dataframe))
            )
            sampled_df = dataframe.sample(n=max_sample_size, random_state=42)

            # Train-test split
            test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
            random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)
            X_train, X_test, Y_train, Y_test = train_test_split(
                sampled_df[feature_columns], sampled_df[selected_target_columns], 
                test_size=test_size, random_state=random_seed
            )

            # Allow user to select models to evaluate
            available_models = {
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_seed),
                "Elastic Net": MultiOutputRegressor(ElasticNet(random_state=random_seed)),
                "AdaBoost Regressor": MultiOutputRegressor(AdaBoostRegressor(random_state=random_seed)),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                "Lasso Regression": MultiOutputRegressor(Lasso(random_state=random_seed)),
                "Ridge Regression": MultiOutputRegressor(Ridge(random_state=random_seed)),
                "Linear Regression": LinearRegression(),
                "MLP Regressor": MultiOutputRegressor(MLPRegressor(random_state=random_seed, max_iter=500)),
                "Random Forest Regressor": RandomForestRegressor(random_state=random_seed),
                "Support Vector Regressor (SVR)": MultiOutputRegressor(SVR()),
            }

            selected_models = st.multiselect(
                "Select Models to Evaluate",
                options=list(available_models.keys()),
                default=list(available_models.keys())
            )

            models = {name: available_models[name] for name in selected_models}

            # Evaluate models
            results = []
            with st.spinner("Evaluating models..."):
                for model_name, model in models.items():
                    try:
                        mae = -cross_val_score(
                            model, X_train, Y_train, scoring="neg_mean_absolute_error", 
                            cv=KFold(n_splits=3, shuffle=True, random_state=random_seed),
                            n_jobs=-1
                        ).mean()
                        results.append({"Model": model_name, "Mean Absolute Error (MAE)": mae})
                    except Exception as e:
                        st.error(f"Error evaluating {model_name}: {e}")

            # Create a DataFrame for results sorted in descending order
            results_df = pd.DataFrame(results).sort_values(by="Mean Absolute Error (MAE)", ascending=True)

            # Display results as a table
            st.write("### Regression Model Performance Comparison")
            st.dataframe(results_df)

            # Bar graph with different colors
            st.write("### Model Performance Bar Chart")
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))  # Generate different colors
            plt.barh(results_df["Model"], results_df["Mean Absolute Error (MAE)"], color=colors)
            plt.xlabel("Mean Absolute Error (MAE)")
            plt.ylabel("Model")
            plt.title("Model Performance Comparison (Bar Chart)")
            plt.gca().invert_yaxis()
            st.pyplot(plt)

            # Truncated model names for the line graph
            truncated_names = [name if len(name) <= 20 else name[:17] + "..." for name in results_df["Model"]]

            # Line graph
            st.write("### Model Performance Line Chart")
            plt.figure(figsize=(10, 6))
            plt.plot(truncated_names, results_df["Mean Absolute Error (MAE)"], marker='o', linestyle='-', color='blue', label="MAE")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Model")
            plt.ylabel("Mean Absolute Error (MAE)")
            plt.title("Model Performance Comparison (Line Chart)")
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)

            # Performance heatmap
            st.write("### Model Performance Heatmap")
            pivot_table = results_df.pivot_table(values="Mean Absolute Error (MAE)", index="Model")
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_table, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", linewidths=0.5)
            plt.title("Model Performance Heatmap")
            st.pyplot(plt)

else:
    st.write("Please upload a dataset.")
