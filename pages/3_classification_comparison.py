import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
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
st.title("ML Model Comparison")

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load the dataset
    dataframe = read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataframe.head())

    # Select features and target
    columns = list(dataframe.columns)
    target_column = st.selectbox("Select the target column", columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in columns if col != target_column])

    if target_column and feature_columns:
        X = dataframe[feature_columns].values
        Y = dataframe[target_column].values

        # Train-test split
        test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.5, 0.2)
        random_seed = st.sidebar.slider("Random Seed", 1, 100, 42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        # Define models
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
            "Gaussian Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=random_seed),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=200, random_state=random_seed),
            "MLP Classifier": MLPClassifier(max_iter=200, random_state=random_seed),
            "Perceptron": Perceptron(random_state=random_seed),
            "Random Forest": RandomForestClassifier(random_state=random_seed),
            "Support Vector Machine (SVM)": SVC(random_state=random_seed),
        }

        # Evaluate models
        results = []
        for model_name, model in models.items():
            model.fit(X_train, Y_train)
            accuracy = model.score(X_test, Y_test)
            results.append({"Model": model_name, "Accuracy": accuracy * 100})

        # Create a DataFrame for results
        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

        # Display results as a table
        st.write("### Model Performance Comparison")
        st.dataframe(results_df)

        # Bar graph with different colors
        st.write("### Model Accuracy Bar Chart")
        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))  # Generate different colors
        plt.barh(results_df["Model"], results_df["Accuracy"], color=colors)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Model")
        plt.title("Model Performance Comparison (Bar Chart)")
        plt.gca().invert_yaxis()
        st.pyplot(plt)

        # Truncated model names for the line graph
        truncated_names = [name if len(name) <= 10 else name[:10] + "..." for name in results_df["Model"]]

        # Line graph
        st.write("### Model Accuracy Line Chart")
        plt.figure(figsize=(10, 6))
        plt.plot(truncated_names, results_df["Accuracy"], marker='o', linestyle='-', color='blue', label="Accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Model")
        plt.ylabel("Accuracy (%)")
        plt.title("Model Performance Comparison (Line Chart)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Performance heatmap
        st.write("### Model Performance Heatmap")
        pivot_table = results_df.pivot_table(values="Accuracy", index="Model")
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="coolwarm", cbar=True, fmt=".2f", linewidths=0.5)
        plt.title("Model Performance Heatmap")
        st.pyplot(plt)

    else:
        st.write("Please select the target and feature columns.")
else:
    st.write("Please upload a dataset.")
