import os
import requests
from google.cloud import run_v2
import streamlit as st

@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlopsdat/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    return os.environ.get("BACKEND", None)

def classify_image(uploaded_file, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    
    # Prepare file tuple: (filename, file object, content type)
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),  # or uploaded_file.read() if not cached
            uploaded_file.type or "image/jpeg"
        )
    }

    response = requests.post(predict_url, files=files, timeout=60)
    if response.status_code == 200:
        return response.json()

    st.write("Response status:", response.status_code)
    st.write("Response content:", response.text)
    return None


def main() -> None:
    backend = get_backend_url()
    if backend is None:
        st.error("Backend service not found.")
        return

    st.title("Image-Based Flower Classification via ResNet-18 and PyTorch")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # assign result inside this block
        result = classify_image(uploaded_file, backend=backend)

        if result is not None:
            prediction = result["Prediction"]
            col1, spacer , col2 = st.columns([3,1, 4]) 
            with col1:
                st.image(uploaded_file.getvalue(), caption="Uploaded Image", width=300)
            with col2:
                st.markdown(f"### Predicted Flower Name: ***{prediction}***",width=400)
        
        else:
            st.error("Failed to get prediction from backend.")
            # You can print extra info here if you have access to response

if __name__ == "__main__":
    main()
