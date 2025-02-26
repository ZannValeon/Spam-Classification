import streamlit as st
import joblib

# 1. Load the trained model
model = joblib.load("spam_detection_model.joblib")

# 2. Set up the Streamlit app title and description
st.title("SMS Spam Detection App")
st.write("Enter a message below, and the model will predict whether it's **Spam** or **Not Spam**.")

# 3. Create a text area for user input
user_message = st.text_area("Message to classify", "")

# 4. When the user clicks the "Predict" button
if st.button("Predict"):
    # The model pipeline includes preprocessing, so we can directly pass the raw text
    prediction = model.predict([user_message])[0]
    probabilities = model.predict_proba([user_message])[0]

    spam_probability = probabilities[1]
    not_spam_probability = probabilities[0]

    # 5. Display the result
    if prediction == 1:
        st.markdown("**Prediction**: Spam")
    else:
        st.markdown("**Prediction**: Not Spam")

    st.write(f"**Spam Probability**: {spam_probability:.2f}")
    st.write(f"**Not-Spam Probability**: {not_spam_probability:.2f}")
