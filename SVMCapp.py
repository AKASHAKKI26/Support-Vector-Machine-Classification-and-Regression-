import pandas as pd
import streamlit as st
import pickle

# Load trained SVM model
with open('svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Prediction function
def predict_svm(input_data):

    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = svm_model.predict(input_df)

    return prediction[0]

# Streamlit app
def main():

    st.title("Titanic Survival Prediction using SVM")

    st.write("Enter passenger details")

    # Input fields
    pclass = st.number_input("Passenger Class", min_value=1, max_value=3)

    sex = st.selectbox("Sex", ["Male", "Female"])

    age = st.number_input("Age", min_value=0)

    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0)

    parch = st.number_input("Parents/Children Aboard", min_value=0)

    fare = st.number_input("Fare", min_value=0.0)

    # Convert categorical value
    sex = 1 if sex == "Male" else 0

    # Prediction button
    if st.button("Predict"):

        input_data = {
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare
        }

        prediction = predict_svm(input_data)

        if prediction == 1:
            st.success("Passenger Survived")
        else:
            st.error("Passenger Did Not Survive")

# Run app
if __name__ == "__main__":
    main()