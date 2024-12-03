import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import os
import joblib
import pandas as pd

# Streamlit app title
st.title("Regression Model Training and Saving")

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
        X = dataframe[feature_columns].values
        Y = dataframe[target_column].values

        # Sidebar for model selection
        model_choice = st.sidebar.selectbox(
            "Select Regression Model", 
            ["Decision Tree Regressor", "Elastic Net", "AdaBoost Regressor", 
             "K-Nearest Neighbors Regressor", "Lasso Regression", "Ridge Regression", 
             "Linear Regression", "MLP Regressor", "Random Forest Regressor", 
             "Support Vector Regressor (SVR)"]
        )

        # Set up hyperparameters based on model choice
        if model_choice == "Decision Tree Regressor":
            st.subheader("Decision Tree Regressor")
            with st.sidebar.expander("Decision Tree Hyperparameters", expanded=True):
                max_depth = st.slider("Max Depth", 1, 50, None, key="dt_max_depth")
                min_samples_split = st.slider("Min Samples Split", 2, 20, 2, key="dt_min_samples_split")
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, key="dt_min_samples_leaf")
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )

        elif model_choice == "Elastic Net":
            st.subheader("Elastic Net Regressor")
            with st.sidebar.expander("Elastic Net Hyperparameters", expanded=True):
                alpha = st.slider("Alpha", 0.01, 10.0, 1.0, key="en_alpha")
                l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, key="en_l1_ratio")
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        elif model_choice == "AdaBoost Regressor":
            st.subheader("AdaBoost Regressor")
            with st.sidebar.expander("AdaBoost Hyperparameters", expanded=True):
                n_estimators = st.slider("Number of Estimators", 10, 500, 50, key="ab_n_estimators")
                learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, key="ab_learning_rate")
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        elif model_choice == "K-Nearest Neighbors Regressor":
            st.subheader("K-Nearest Neighbors Regressor")
            with st.sidebar.expander("KNN Hyperparameters", expanded=True):
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors")
                weights = st.selectbox("Weights", ["uniform", "distance"], key="knn_weights")
                algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key="knn_algorithm")
            model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

        elif model_choice == "Lasso Regression":
            st.subheader("Lasso Regression")
            with st.sidebar.expander("Lasso Hyperparameters", expanded=True):
                alpha = st.slider("Alpha", 0.01, 10.0, 1.0, key="lasso_alpha")
            model = Lasso(alpha=alpha, random_state=42)

        elif model_choice == "Ridge Regression":
            st.subheader("Ridge Regression")
            with st.sidebar.expander("Ridge Hyperparameters", expanded=True):
                alpha = st.slider("Alpha", 0.01, 10.0, 1.0, key="ridge_alpha")
            model = Ridge(alpha=alpha, random_state=42)

        elif model_choice == "Linear Regression":
            st.subheader("Linear Regression")
            model = LinearRegression()

        elif model_choice == "MLP Regressor":
            st.subheader("MLP Regressor")
            with st.sidebar.expander("MLP Hyperparameters", expanded=True):
                hidden_layer_sizes = tuple(map(int, st.text_input("Hidden Layer Sizes (e.g., 100,50)", "100,50").split(',')))
                activation = st.selectbox("Activation Function", ["identity", "logistic", "tanh", "relu"])
                max_iter = st.slider("Max Iterations", 100, 2000, 1000, key="mlp_max_iter")
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, random_state=42)

        elif model_choice == "Random Forest Regressor":
            st.subheader("Random Forest Regressor")
            with st.sidebar.expander("Random Forest Hyperparameters", expanded=True):
                n_estimators = st.slider("Number of Trees", 10, 500, 100, key="rf_n_estimators")
                max_depth = st.slider("Max Depth", 1, 50, None, key="rf_max_depth")
                min_samples_split = st.slider("Min Samples Split", 2, 10, 2, key="rf_min_samples_split")
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, key="rf_min_samples_leaf")
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)

        elif model_choice == "Support Vector Regressor (SVR)":
            st.subheader("Support Vector Regressor (SVR)")
            with st.sidebar.expander("SVR Hyperparameters", expanded=True):
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key="svr_kernel")
                C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key="svr_C")
                epsilon = st.slider("Epsilon", 0.0, 1.0, 0.1, key="svr_epsilon")
            model = SVR(kernel=kernel, C=C, epsilon=epsilon)

        # Split the dataset into training and testing sets
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.slider("Random Seed", 1, 100, 42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        # Train the selected model
        model.fit(X_train, Y_train)

        # Evaluate the model
        mae = -cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=KFold(n_splits=5, shuffle=True, random_state=random_seed)).mean()
        st.write(f"Mean Absolute Error (MAE): {mae:.3f}")

        # Save the model
        if st.button(f"Save {model_choice} Model"):
            save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
            os.makedirs(save_folder, exist_ok=True)
            model_filename = os.path.join(save_folder, f"regression_{model_choice.lower().replace(' ', '_')}_model.joblib")
            joblib.dump(model, model_filename)
            st.success(f"Model saved at: {model_filename}")
    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a dataset.")
