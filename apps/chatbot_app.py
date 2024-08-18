import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the model and tokenizer
model_path = 'models/Chatbot/Python/Models/Models_local/local_medical_assistant_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def app():
    st.title("Medical Assistant Chatbot")

    # User input field
    input_text = st.text_area("Ask your question:", "")

    # Button to generate response
    if st.button("Reply"):
        if input_text.strip() != "":
            # Tokenize the input
            inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

            # Set the model to evaluation mode
            model.eval()

            # Generate the response
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=512,       # Maximum response length
                    num_beams=2,          # Beam search
                    early_stopping=True,  # Early stopping
                    no_repeat_ngram_size=1 # Prevent repetition
                )

            # Decode the response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write(f"Doctor AI: {generated_text}")
        else:
            st.warning("Please ask a question.")
