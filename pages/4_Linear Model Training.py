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
import numpy as np

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Train Model", "Air Quality Predictor"])

if selection == "Train Model":
    st.title("Regression Model Training and Air Quality Prediction")

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

        # Define target columns
        target_columns = ["T", "AH", "RH", "CO_level"]
        available_targets = [col for col in target_columns if col in dataframe.columns]

        if not available_targets:
            st.warning("No expected target columns (T, AH, RH, CO_level) found in the dataset.")
        else:
            # Allow user to select target columns
            selected_target_columns = st.multiselect(
                "Select target columns (Y):",
                available_targets,
                default=available_targets
            )

            # Allow user to select feature columns, excluding the target columns
            feature_columns = st.multiselect(
                "Select feature columns (X):",
                dataframe.columns.tolist(),
                [col for col in dataframe.columns if col not in selected_target_columns]
            )

            if selected_target_columns and feature_columns:
                # Extract features (X) and targets (Y) from the dataset
                X = dataframe[feature_columns].values
                Y = dataframe[selected_target_columns].values

                # Sidebar for model selection
                model_choice = st.sidebar.selectbox(
                    "Select Regression Model",
                    [
                        "Decision Tree Regressor", "Elastic Net", "AdaBoost Regressor",
                        "K-Nearest Neighbors Regressor", "Lasso Regression", "Ridge Regression",
                        "Linear Regression", "MLP Regressor", "Random Forest Regressor",
                        "Support Vector Regressor (SVR)"
                    ]
                )

                # Model setup
                if model_choice == "Decision Tree Regressor":
                    st.subheader("Decision Tree Regressor")
                    with st.sidebar.expander("Decision Tree Hyperparameters"):
                        max_depth = st.slider("Max Depth", 1, 50, 10)
                        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
                    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

                elif model_choice == "Elastic Net":
                    st.subheader("Elastic Net Regressor")
                    with st.sidebar.expander("Elastic Net Hyperparameters"):
                        alpha = st.slider("Alpha", 0.01, 10.0, 1.0)
                        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

                elif model_choice == "AdaBoost Regressor":
                    st.subheader("AdaBoost Regressor")
                    with st.sidebar.expander("AdaBoost Hyperparameters"):
                        n_estimators = st.slider("Number of Estimators", 10, 500, 50)
                        learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0)
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

                # Train/Test Split
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
                random_seed = st.slider("Random Seed", 1, 100, 42)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

                # Train the model
                model.fit(X_train, Y_train)

                # Evaluate the model
                mae = -cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=KFold(n_splits=5, shuffle=True, random_state=random_seed)).mean()
                st.write(f"Mean Absolute Error (MAE): {mae:.3f}")

                # Save the model and feature names
                if st.button(f"Save {model_choice} Model"):
                    save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                    os.makedirs(save_folder, exist_ok=True)
                    model_filename = os.path.join(save_folder, f"{model_choice.replace(' ', '_')}_model.joblib")
                    joblib.dump(model, model_filename)
                    st.success(f"Model saved at: {save_folder}")
    else:
        st.info("Please upload a CSV file to begin training.")


elif selection == "Air Quality Predictor":
    st.header("Air Quality Predictor")

    # File uploader for the trained model
    uploaded_model = st.file_uploader("Upload a trained model (.joblib)", type="joblib")

    if uploaded_model is not None:
        # Load the custom uploaded model
        model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")

        # Input form for new predictions
        # Layout for Date and Time inputs using columns
        col1, col2, col3 = st.columns(3)

        with col1:
            date = st.date_input("Enter Date", value=None)

        with col2:
            time = st.time_input("Enter Time", value=None)

        with col3:
            CO_GT = st.number_input("Enter CO_GT (mg/m³)", min_value=0.0)

        # Layout for other features (3 features per row)
        col3, col4, col5 = st.columns(3)
        with col3:
            PT08_S5_O3 = st.number_input("Enter PT08_S5_O3", min_value=0.0)
        with col4:
            NMHC_GT = st.number_input("Enter NMHC_GT (µg/m³)", min_value=0.0)
        with col5:
            C6H6_GT = st.number_input("Enter C6H6_GT (µg/m³)", min_value=0.0)

        col6, col7, col8 = st.columns(3)
        with col6:
            PT08_S1_CO = st.number_input("Enter PT08_S1_CO (sensor value)", min_value=0.0)
        with col7:
            Nox_GT = st.number_input("Enter Nox_GT (ppb)", min_value=0.0)
        with col8:
            PT08_S2_NMHC = st.number_input("Enter PT08_S2_NMHC (sensor value)", min_value=0.0)

        col9, col10, col11 = st.columns(3)
        with col9:
            NO2_GT = st.number_input("Enter NO2_GT (ppb)", min_value=0.0)
        with col10:
            PT08_S3_Nox = st.number_input("Enter PT08_S3_Nox (sensor value)", min_value=0.0)
        with col11:
            PT08_S4_NO2 = st.number_input("Enter PT08_S4_NO2 (sensor value)", min_value=0.0)

        # Prepare input data
        input_data = np.array([
            CO_GT, PT08_S1_CO, NMHC_GT, C6H6_GT, PT08_S2_NMHC,
            Nox_GT, PT08_S3_Nox, NO2_GT, PT08_S4_NO2, PT08_S5_O3
        ]).reshape(1, -1)

        # Perform the prediction
        if st.button("Predict"):
            try:
                # Perform the prediction
                prediction = model.predict(input_data)

                # Check if the model predicts multiple outputs
                if prediction.ndim == 2 and prediction.shape[1] == 4:
                    # Unpack predictions
                    predicted_T, predicted_RH, predicted_AH, predicted_CO_level = prediction[0]

                    # Show predictions
                    st.write(f"Predicted Temperature (T): {predicted_T:.2f} °C")
                    st.write(f"Predicted Relative Humidity (RH): {predicted_RH:.2f} %")
                    st.write(f"Predicted Absolute Humidity (AH): {predicted_AH:.2f} g/m³")
                    st.write(f"Predicted CO Level: {predicted_CO_level:.2f}")
                else:
                    st.write(f"Prediction: {prediction[0]:.2f}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    else:
        st.info("Please upload a trained model file to make predictions.")

else:
    st.warning("Invalid selection. Please use the sidebar to navigate.")