import streamlit as st
from pandas import read_csv, get_dummies
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
import joblib

# Streamlit app title
st.title("ML Model Training and Saving")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    dataframe = read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataframe.head())

    # Convert categorical variables to numerical (One-Hot Encoding)
    dataframe = get_dummies(dataframe, drop_first=True)
    st.write("Processed Dataset:")
    st.write(dataframe.head())

    # Select features and target
    columns = list(dataframe.columns)
    target_column = st.selectbox("Select the target column", columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in columns if col != target_column])

    if target_column and feature_columns:
        X = dataframe[feature_columns].values
        Y = dataframe[target_column].values

        # Sidebar for model selection
        model_choice = st.sidebar.selectbox(
            "Select Machine Learning Model", 
            ["Decision Tree", "Gaussian Naive Bayes", "AdaBoost", "K-Nearest Neighbors", 
             "Logistic Regression", "MLP Classifier", "Perceptron Classifier", "Random Forest", 
             "Support Vector Machine (SVM)"]
        )

        if model_choice == "Decision Tree":
            st.subheader("Decision Tree Classifier")

            # Decision Tree Hyperparameters
            with st.sidebar.expander("Decision Tree Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="dt_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 50, key="dt_random_seed")
                max_depth = st.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
                min_samples_split = st.slider("Min Samples Split", 2, 10, 2, key="dt_min_samples_split")
                min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, key="dt_min_samples_leaf")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train Decision Tree
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_seed
            )
            model.fit(X_train, Y_train)

            # Evaluate Decision Tree
            accuracy = model.score(X_test, Y_test)
            st.write(f"Decision Tree Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save Decision Tree Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_decision_tree_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "Gaussian Naive Bayes":
            st.subheader("Gaussian Naive Bayes Classifier")

            # Gaussian Naive Bayes Hyperparameters
            with st.sidebar.expander("Gaussian Naive Bayes Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="gnb_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="gnb_random_seed")
                var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=-9, step=1, key="gnb_var_smoothing")

            # Convert var_smoothing from log scale to regular scale
            var_smoothing_value = 10 ** var_smoothing

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train Gaussian Naive Bayes
            model = GaussianNB(var_smoothing=var_smoothing_value)
            model.fit(X_train, Y_train)

            # Evaluate Gaussian Naive Bayes
            accuracy = model.score(X_test, Y_test)
            st.write(f"Gaussian Naive Bayes Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save Gaussian Naive Bayes Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_gaussian_nb_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "AdaBoost":
            st.subheader("AdaBoost Classifier")

            # AdaBoost Hyperparameters
            with st.sidebar.expander("AdaBoost Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="ada_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="ada_random_seed")
                n_estimators = st.slider("Number of Estimators", 1, 100, 50, key="ada_n_estimators")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train AdaBoost
            model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)
            model.fit(X_train, Y_train)

            # Evaluate AdaBoost
            accuracy = model.score(X_test, Y_test)
            st.write(f"AdaBoost Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save AdaBoost Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_adaboost_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "K-Nearest Neighbors":
            st.subheader("K-Nearest Neighbors Classifier")

            # KNN Hyperparameters
            with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="knn_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="knn_random_seed")
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors")
                weights = st.selectbox("Weights", options=["uniform", "distance"], key="knn_weights")
                algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"], key="knn_algorithm")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train KNN
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
            model.fit(X_train, Y_train)

            # Evaluate KNN
            accuracy = model.score(X_test, Y_test)
            st.write(f"K-Nearest Neighbors Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save KNN Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_knn_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "Logistic Regression":
            st.subheader("Logistic Regression Classifier")

            # Logistic Regression Hyperparameters
            with st.sidebar.expander("Logistic Regression Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="lr_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="lr_random_seed")
                max_iter = st.slider("Max Iterations", 100, 500, 200, key="lr_max_iter")
                solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"], key="lr_solver")
                C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=1.0, key="lr_C")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train Logistic Regression
            model = LogisticRegression(max_iter=max_iter, solver=solver, C=C, random_state=random_seed)
            model.fit(X_train, Y_train)

            # Evaluate Logistic Regression
            accuracy = model.score(X_test, Y_test)
            st.write(f"Logistic Regression Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save Logistic Regression Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_logistic_regression_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "MLP Classifier":
            st.subheader("MLP Classifier")

            # MLP Classifier Hyperparameters
            with st.sidebar.expander("MLP Classifier Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="mlp_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="mlp_random_seed")
                hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 100,50)", "100,50", key="mlp_hidden_layers")
                activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"], key="mlp_activation")
                max_iter = st.slider("Max Iterations", 100, 500, 200, key="mlp_max_iter")

            # Convert hidden_layer_sizes input to tuple
            hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train MLP Classifier
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver='adam',
                max_iter=max_iter,
                random_state=random_seed
            )
            model.fit(X_train, Y_train)

            # Evaluate MLP Classifier
            accuracy = model.score(X_test, Y_test)
            st.write(f"MLP Classifier Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save MLP Classifier Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_mlp_classifier_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "Perceptron Classifier":
            st.subheader("Perceptron Classifier")

            # Perceptron Classifier Hyperparameters
            with st.sidebar.expander("Perceptron Classifier Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="perceptron_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="perceptron_random_seed")
                max_iter = st.slider("Max Iterations", 100, 500, 200, key="perceptron_max_iter")
                eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=1.0, key="perceptron_eta0")
                tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=1e-3, key="perceptron_tol")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train Perceptron Classifier
            model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
            model.fit(X_train, Y_train)

            # Evaluate Perceptron Classifier
            accuracy = model.score(X_test, Y_test)
            st.write(f"Perceptron Classifier Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save Perceptron Classifier Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_perceptron_classifier_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "Random Forest":
            st.subheader("Random Forest Classifier")

            # Random Forest Hyperparameters
            with st.sidebar.expander("Random Forest Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="rf_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 7, key="rf_random_seed")
                n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, 100, key="rf_n_estimators")
                max_depth = st.slider("Max Depth of Trees", 1, 50, None, key="rf_max_depth")  # Allows None for no limit
                min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, 2, key="rf_min_samples_split")
                min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, 1, key="rf_min_samples_leaf")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train Random Forest Classifier
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_seed,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            model.fit(X_train, Y_train)

            # Evaluate Random Forest Classifier
            accuracy = model.score(X_test, Y_test)
            st.write(f"Random Forest Classifier Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save Random Forest Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_random_forest_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")

        elif model_choice == "Support Vector Machine (SVM)":
            st.subheader("Support Vector Machine (SVM)")

            # SVM Hyperparameters
            with st.sidebar.expander("Support Vector Machine (SVM) Hyperparameters", expanded=True):
                test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, key="svm_test_size")
                random_seed = st.slider("Random Seed", 1, 100, 42, key="svm_random_seed")
                C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, key="svm_C")
                kernel = st.selectbox("Kernel Type", options=["linear", "poly", "rbf", "sigmoid"], key="svm_kernel")

            # Split dataset
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

            # Train SVM Classifier
            model = SVC(kernel=kernel, C=C, random_state=random_seed)
            model.fit(X_train, Y_train)

            # Evaluate SVM Classifier
            accuracy = model.score(X_test, Y_test)
            st.write(f"Support Vector Machine (SVM) Accuracy: {accuracy * 100.0:.3f}%")

            # Save the model
            if st.button("Save SVM Model"):
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                model_filename = os.path.join(save_folder, "classification_svm_model.joblib")
                joblib.dump(model, model_filename)
                st.success(f"Model saved at: {model_filename}")
    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a CSV file.")


    