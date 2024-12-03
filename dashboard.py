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
st.title("ML Model Training and Lung Cancer Prediction")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose Section", ["Train Model", "Make Predictions"])

if section == "Train Model":
    st.header("Train a Classification Model")

    # Upload the dataset
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load the dataset
        dataframe = read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(dataframe.head())

        # Convert columns from Smoking to Chest Pain (2 = 1, 1 = 0)
        columns_to_convert = ["SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"]
        if all(col in dataframe.columns for col in columns_to_convert):
            for col in columns_to_convert:
                dataframe[col] = dataframe[col].replace({2: 1, 1: 0})
            st.write("Processed Dataset (Converted Smoking to Chest Pain Columns):")
            st.write(dataframe.head())
        else:
            missing_columns = [col for col in columns_to_convert if col not in dataframe.columns]
            st.warning(f"The following required columns are missing from the dataset: {missing_columns}")

        # One-Hot Encoding for categorical features
        dataframe = get_dummies(dataframe, drop_first=True)
        st.write("Processed Dataset (After Encoding):")
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
                max_depth = st.slider("Max Depth", 1, 50, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

            elif model_choice == "Gaussian Naive Bayes":
                st.subheader("Gaussian Naive Bayes Classifier")
                var_smoothing = st.number_input("Var Smoothing", 1e-9, 1e-5, 1e-8, format="%.1e")
                model = GaussianNB(var_smoothing=var_smoothing)

            elif model_choice == "AdaBoost":
                st.subheader("AdaBoost Classifier")
                n_estimators = st.slider("Number of Estimators", 50, 500, 100)
                model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "K-Nearest Neighbors":
                st.subheader("K-Nearest Neighbors Classifier")
                n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)

            elif model_choice == "Logistic Regression":
                st.subheader("Logistic Regression Classifier")
                C = st.slider("Inverse of Regularization Strength (C)", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, max_iter=200, random_state=42)

            elif model_choice == "MLP Classifier":
                st.subheader("MLP Classifier")
                hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 100,50)", "100,50")
                hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(",")))
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=200, random_state=42)

            elif model_choice == "Perceptron Classifier":
                st.subheader("Perceptron Classifier")
                eta0 = st.slider("Learning Rate (eta0)", 0.01, 1.0, 0.1)
                model = Perceptron(eta0=eta0, random_state=42)

            elif model_choice == "Random Forest":
                st.subheader("Random Forest Classifier")
                n_estimators = st.slider("Number of Trees", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "Support Vector Machine (SVM)":
                st.subheader("Support Vector Machine (SVM)")
                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                C = st.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0)
                model = SVC(kernel=kernel, C=C, random_state=42)

             # Train the model
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

            if st.button("Train Model"):
                model.fit(X_train, Y_train)
                accuracy = model.score(X_test, Y_test)
                st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

                # Save the model
                save_folder = r'C:\Users\user\Desktop\jeah\ITD105\LAB3\Models'
                os.makedirs(save_folder, exist_ok=True)
                save_path = os.path.join(save_folder, f"classification_{model_choice.replace(' ', '_').lower()}_model.joblib")
                joblib.dump(model, save_path)
                st.success(f"Model saved as: {save_path}")

elif section == "Make Predictions":
    st.header("Lung Cancer Diagnosis Predictor")
    uploaded_model = st.file_uploader("Upload a Trained Model (joblib format)", type=["joblib"])
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("Model successfully loaded!")

        st.subheader("Input Sample Data for Prediction")

        # Input Fields for Prediction
        def convert_yes_no(value):
            return 1 if value == "Yes" else 0  # Adjusted for the new mapping (Yes=1, No=0)

        # Input features (15 expected by the model), following the specified order
        # Row 1: Gender, Age, Smoking
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            gender = 1 if gender == "Male" else 0  # Convert "Male" to 1 and "Female" to 0
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
        with col3:
            smoking = convert_yes_no(st.selectbox("Smoking", ["Yes", "No"]))

        # Row 2: Anxiety, Alcohol, Chronic Disease, Chest Pain
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            anxiety = convert_yes_no(st.selectbox("Anxiety", ["Yes", "No"]))
        with col2:
            alcohol_consumption = convert_yes_no(st.selectbox("Alcohol Consumption", ["Yes", "No"]))
        with col3:
            chronic_disease = convert_yes_no(st.selectbox("Chronic Disease", ["Yes", "No"]))
        with col4:
            chest_pain = convert_yes_no(st.selectbox("Chest Pain", ["Yes", "No"]))

        # Row 3: Peer Pressure, Allergy, Wheezing, Fatigue
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            peer_pressure = convert_yes_no(st.selectbox("Peer Pressure", ["Yes", "No"]))
        with col2:
            allergy = convert_yes_no(st.selectbox("Allergy", ["Yes", "No"]))
        with col3:
            wheezing = convert_yes_no(st.selectbox("Wheezing", ["Yes", "No"]))
        with col4:
            fatigue = convert_yes_no(st.selectbox("Fatigue", ["Yes", "No"]))

        # Row 4: Coughing, Shortness of Breath, Difficulty Swallowing, Yellow Fingers
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            coughing = convert_yes_no(st.selectbox("Coughing", ["Yes", "No"]))
        with col2:
            shortness_of_breath = convert_yes_no(st.selectbox("Shortness of Breath", ["Yes", "No"]))
        with col3:
            swallowing_difficulty = convert_yes_no(st.selectbox("Swallowing Difficulty", ["Yes", "No"]))
        with col4:
            yellow_fingers = convert_yes_no(st.selectbox("Yellow Fingers", ["Yes", "No"]))

        # Combine inputs
        input_data = [
            gender, age, smoking, anxiety, alcohol_consumption, chronic_disease, chest_pain,
            peer_pressure, allergy, wheezing, fatigue, coughing, shortness_of_breath,
            swallowing_difficulty, yellow_fingers
        ]

        # Predict using the loaded model
        if st.button("Predict"):
            prediction = model.predict([input_data])
            if prediction[0] == 1:
                result = '<span style="color:green;">Positive for Lung Cancer</span>'
            else:
                result = '<span style="color:red;">Negative for Lung Cancer</span>'
            
            st.markdown(f"### Prediction: {result}", unsafe_allow_html=True)
