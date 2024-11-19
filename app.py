import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Define the Streamlit app
st.title("Weighted Compatibility Scoring App")
st.write("Upload your dataset, select two rows, and adjust weights to find a personalized similarity score!")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(dataset)

    # Ensure necessary columns exist
    required_columns = ["How would you describe your personality?", "Movie Genre",
                        "Music Taste", "What qualities do you value most in a partner?"]
    if all(col in dataset.columns for col in required_columns):
        
        # Step 2: Sidebar for weights
        st.sidebar.header("Feature Weights")
        weight_personality = st.sidebar.slider("Weight: Personality", 0.0, 1.0, 0.25, 0.05)
        weight_movie = st.sidebar.slider("Weight: Movie Genre", 0.0, 1.0, 0.25, 0.05)
        weight_music = st.sidebar.slider("Weight: Music Taste", 0.0, 1.0, 0.25, 0.05)
        weight_qualities = st.sidebar.slider("Weight: Partner Qualities", 0.0, 1.0, 0.25, 0.05)
        
        # Normalize weights to ensure they sum to 1
        total_weight = weight_personality + weight_movie + weight_music + weight_qualities
        if total_weight > 0:
            weight_personality /= total_weight
            weight_movie /= total_weight
            weight_music /= total_weight
            weight_qualities /= total_weight

        # Step 3: Allow user to select two rows for compatibility scoring
        st.write("Select two rows to compare:")
        row1 = st.number_input("Row 1 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
        row2 = st.number_input("Row 2 (Index)", min_value=0, max_value=len(dataset)-1, step=1)
        
        if row1 != row2:
            # Step 4: Compute weighted similarity between the selected rows
            def compute_weighted_similarity(row1, row2):
                user1 = dataset.loc[row1, required_columns]
                user2 = dataset.loc[row2, required_columns]
                
                # Raw similarities for each feature
                similarity_personality = 1 if user1["How would you describe your personality?"] == user2["How would you describe your personality?"] else 0
                similarity_movie = 1 if user1["Movie Genre"] == user2["Movie Genre"] else 0
                similarity_music = 1 if user1["Music Taste"] == user2["Music Taste"] else 0
                similarity_qualities = 1 if user1["What qualities do you value most in a partner?"] == user2["What qualities do you value most in a partner?"] else 0
                
                # Weighted similarity
                weighted_similarity = (
                    weight_personality * similarity_personality +
                    weight_movie * similarity_movie +
                    weight_music * similarity_music +
                    weight_qualities * similarity_qualities
                )
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
        st.error(f"The dataset must contain the following columns: {required_columns}")
else:
    st.write("Awaiting CSV file upload...")
