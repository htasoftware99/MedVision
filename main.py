import streamlit as st
from streamlit_option_menu import option_menu

# Import the app modules
from apps.brain_tumor_app import app as brain_tumor_app
from apps.breast_app import app as breast_app
from apps.chatbot_app import app as chatbot_app
from apps.chest_app import app as chest_app
from apps.diabete_app import app as diabetes_app
from apps.heart_app import app as heart_app
from apps.covid_app import app as covid_app

def main():
    # Create a sidebar menu with icons
    with st.sidebar:
        app_selection = option_menu(
            menu_title=None,  # No menu title
            options=[
                "Brain Tumor Detection", 
                "Breast Cancer Prediction", 
                "Medical Assistant Chatbot",
                "Pneumonia - Normal X-ray", 
                "Diabetes Prediction", 
                "Heart Disease Prediction",
                "Covid Detection"
            ],
            icons=[
                "bi bi-radioactive",  # Brain icon
                "bi bi-house-heart",  # Heart icon
                "bi bi-chat",  # Chat icon
                "bi bi-x-diamond",  # X-ray icon
                "bi bi-file-earmark-medical",  # Medical file icon
                "bi bi-heart-pulse",
                "bi bi-virus"
            ],
            default_index=0,  # Default selected item
            orientation="vertical"  # Menu orientation
        )
    
    # Handle the selected application
    if app_selection == "Brain Tumor Detection":
        brain_tumor_app()
    elif app_selection == "Breast Cancer Prediction":
        breast_app()
    elif app_selection == "Medical Assistant Chatbot":
        chatbot_app()
    elif app_selection == "Pneumonia - Normal X-ray":
        chest_app()
    elif app_selection == "Diabetes Prediction":
        diabetes_app()
    elif app_selection == "Heart Disease Prediction":
        heart_app()
    elif app_selection == "Covid Detection":
        covid_app()

if __name__ == "__main__":
    main()
