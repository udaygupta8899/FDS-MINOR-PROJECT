import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall

# Step 1: Define the Streamlit app
st.title("Weighted Compatibility Scoring App with Evaluation Metrics")
st.write("Upload your dataset, select two rows, and adjust weights to find a personalized similarity score and evaluate the model!")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(dataset)

    # Display features dynamically in the sidebar
    st.sidebar.header("Feature Weights")
    weight_dict = {}

    # Step 2: Sidebar for weights for each feature dynamically
    for col in dataset.columns:
        if dataset[col].dtype != 'object':  # Skip non-numeric columns for simplicity
            continue
        weight_dict[col] = st.sidebar.slider(f"Weight: {col}", 0.0, 1.0, 0.25, 0.05)
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weight_dict.values())
    if total_weight > 0:
        for col in weight_dict:
            weight_dict[col] /= total_weight

    # Step 3: Allow user to select two rows for compatibility scoring
    st.write("Select two rows to compare:")
    row1 = st.number_input("Row 1 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
    row2 = st.number_input("Row 2 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
    
    if row1 != row2:
        # Step 4: Compute weighted similarity between the selected rows
        def compute_weighted_similarity(row1, row2):
            user1 = dataset.loc[row1, dataset.columns]
            user2 = dataset.loc[row2, dataset.columns]
            
            # Raw similarities for each feature
            similarity = 0
            for col in weight_dict:
                similarity += weight_dict[col] * (1 if user1[col] == user2[col] else 0)
                
            return similarity * 100  # Convert to percentage
            
        weighted_similarity = compute_weighted_similarity(row1, row2)
        st.write(f"Weighted Similarity between Row {row1} and Row {row2}: **{weighted_similarity:.2f}%**")

        # Step 5: Build and train a model for advanced scoring
        def build_model(input_shape):
            model = Sequential([
                Dense(8, activation='relu', input_shape=input_shape),
                Dense(4, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
            return model
        
        # Convert dataset into a binary classification (for demonstration)
        # Here we create a synthetic target variable (binary classification)
        X = dataset[dataset.columns].values  # All columns as features
        y = (np.random.rand(len(dataset)) > 0.5).astype(int)  # Random binary target for demonstration

        # Split dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Build and train the model
        model = build_model((X_train.shape[1],))
        model.fit(X_train, y_train, epochs=10, verbose=0)

        # Step 6: Evaluate the model
        loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1)
        st.write(f"Test Loss: {loss:.4f}")
        st.write(f"Test Accuracy: {accuracy:.4f}")
        st.write(f"Test Precision: {precision:.4f}")
        st.write(f"Test Recall: {recall:.4f}")

        # Step 7: Calculate additional metrics (F1-Score)
        y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary labels
        f1 = f1_score(y_test, y_pred)
        st.write(f"Test F1 Score: {f1:.4f}")

    else:
        st.write("Please select two different rows.")
else:
    st.write("Awaiting CSV file upload...")
