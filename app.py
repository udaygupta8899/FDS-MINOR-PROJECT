import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Define the Streamlit app
st.title("Weighted Compatibility Scoring App")
st.write("Upload your dataset and adjust weights for each feature to find a personalized similarity score!")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(dataset)

    # Step 2: Sidebar for feature weights
    st.sidebar.header("Feature Weights")
    
    # Create sliders for each column in the dataset
    weight_sliders = {}
    for col in dataset.columns:
        weight_sliders[col] = st.sidebar.slider(f"Weight: {col}", 0.0, 1.0, 0.25, 0.05)
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weight_sliders.values())
    if total_weight > 0:
        # Normalize each weight
        for col in weight_sliders:
            weight_sliders[col] /= total_weight

    # Step 3: Allow user to select two rows for compatibility scoring
    st.write("Select two rows to compare:")
    row1 = st.number_input("Row 1 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
    row2 = st.number_input("Row 2 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
    
    if row1 != row2:
        # Step 4: Compute weighted similarity between the selected rows
        def compute_weighted_similarity(row1, row2):
            user1 = dataset.loc[row1]
            user2 = dataset.loc[row2]
            
            weighted_similarity = 0
            
            # Calculate similarity for each feature and apply weight
            for col in dataset.columns:
                # Using exact match for simplicity, can be customized for other similarity measures
                similarity = 1 if user1[col] == user2[col] else 0
                weighted_similarity += weight_sliders[col] * similarity
                
            return weighted_similarity * 100  # Convert to percentage
        
        weighted_similarity = compute_weighted_similarity(row1, row2)
        st.write(f"Weighted Similarity between Row {row1} and Row {row2}: **{weighted_similarity:.2f}%**")

        # Step 5: Build and load a pre-trained model for more advanced scoring
        # Here we'll train the model with a placeholder dataset, but you should load a pre-trained model
        def build_model():
            model = Sequential([
                Dense(8, activation='relu', input_shape=(1,)),
                Dense(4, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        
        # Placeholder for training - in practice, load a pre-trained model
        model = build_model()
        # Train the model on a small synthetic dataset (for demonstration)
        X_train = np.array([[0.1], [0.5], [0.8]])
        y_train = np.array([[0], [1], [1]])
        model.fit(X_train, y_train, epochs=10, verbose=0)
        
        # Predict compatibility score using the trained model
        model_input = np.array([[weighted_similarity / 100]])
        compatibility_score = model.predict(model_input)[0][0] * 100  # Convert to percentage
        st.write(f"Compatibility Score (Predicted): **{compatibility_score:.2f}%**")

    else:
        st.write("Please select two different rows.")
else:
    st.write("Awaiting CSV file upload...")
