import streamlit as st
from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY  # Import your API key from a separate config file
from PIL import Image
import requests
from io import BytesIO

# Streamlit App Configuration
st.set_page_config(page_title="Llama-3.2 Demo App", page_icon="🤖", layout="wide")
st.title("🖼️ Llama-3.2-90B-Vision-Instruct Demo App")
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Enter an image URL and get a description</p>", unsafe_allow_html=True)

# User Inputs with placeholder
image_url = st.text_input("Enter Image URL", value="", placeholder="Paste image URL here...", max_chars=400)
user_prompt = st.text_input("Enter your prompt", value="Describe this image in a paragraph", placeholder="e.g., What is shown in the image?")

# Function to display the image from URL with height limit based on its actual size
def show_image_from_url(image_url, max_height=200):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Get the original image size
        img_width, img_height = img.size

        # Calculate the new height and width based on the max height while maintaining the aspect ratio
        if img_height > max_height:
            aspect_ratio = img_width / img_height
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            img_resized = img.resize((new_width, new_height))
        else:
            img_resized = img  # No resizing needed if the image is smaller than the max height

        # Center the image and display it
        st.image(img_resized, caption=f"Source: {image_url}", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Unable to load image. Error: {e}")

# Process user input
if st.button("Get Description", key="get_description"):
    if image_url and user_prompt:
        try:
            # Show the image with dynamic resizing based on the image size
            show_image_from_url(image_url, max_height=600)

            # Initialize the InferenceClient
            client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

            # Define messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]

            # Call the model
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=messages,
                max_tokens=500
            )

            # Extract JSON response
            model_response = completion.choices[0].message

            # Display the result in a clean and simple format
            st.subheader("📝 Model Response")

            # Display Content
            st.markdown(f"**Description**: {model_response.get('content', 'No description available')}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
    else:
        st.warning("⚠️ Please enter an image URL and a prompt.")

# Clean UI Enhancements
st.markdown("""
    <style>
        .stButton>button {
            background-color: #0072BB;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #005f8a;
        }

        .stTextInput>div>div>input {
            padding: 10px;
            font-size: 16px;
            border-radius: 10px;
        }

        /* Center the image */
        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)