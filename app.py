import os
import sys
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# adding src folder to the path to ensure successful file calling
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model_arch import CustomResNet

# Local function to load test data directly without relying on an external file
def get_test_loader(batch_size=128):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Loading test data directly from torchvision
    test_set = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader

# Basic page settings
st.set_page_config(
    page_title="SmartCatalog - AI Fashion Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the model and its ready weights
@st.cache_resource
def load_app_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomResNet(num_classes=10).to(device)
    weights_path = os.path.join("models", "smart_catalog_resnet.pth")
    
    if os.path.exists(weights_path):
        # Loading the complete file
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Checking if the file is saved as a dictionary containing model_state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, device
    else:
        return None, device
# Loading the model and readiness
model, device = load_app_model()

# Names of the classes for Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Suitable transformations for the input image to match the model's inputs (Grayscale, 28x28)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Main Interface ---
st.title("🛍️ SmartCatalog: AI-Powered Apparel Classification")
st.markdown("This web platform demonstrates real-time inference on fashion items using our custom scratch-built ResNet architecture.")

# Dividing the page into a sidebar and a main display
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.info("System Engine: Custom ResNet\nDataset: Fashion-MNIST\nNo Transfer Learning used.")
    
    st.markdown("---")
    st.subheader("📊 Model Evaluation Hub")
    st.markdown("Evaluate model performance on the independent Test Set instantly.")
    
    # Button to run the evaluation matrix without retraining
    if st.button("📈 Run Model Evaluation"):
        with st.spinner("Calculating performance metrics on Test Set..."):
            try:
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
                import pandas as pd
                
                # Using the safe local loading function
                test_loader = get_test_loader(batch_size=128)
                
                all_preds = []
                all_targets = []
                
                # calculating the predictions on the complete test data
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(labels.cpu().numpy())
                
                # calculating the metrics
                acc = accuracy_score(all_targets, all_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
                cm = confusion_matrix(all_targets, all_preds)
                
                # saving the values in the Session State to display them on the main screen
                st.session_state['evaluation_done'] = True
                st.session_state['eval_metrics'] = {
                    'Accuracy': f"{acc*100:.2f}%",
                    'Precision': f"{prec:.4f}",
                    'Recall': f"{rec:.4f}",
                    'F1-Score': f"{f1:.4f}"
                }
                st.session_state['confusion_matrix'] = cm
                st.success("Evaluation completed successfully!")
            except Exception as e:
                st.error(f"Error executing evaluation: {str(e)}")

# --- display the evaluation output ---
if 'evaluation_done' in st.session_state and st.session_state['evaluation_done']:
    st.markdown("## 📊 Model Test-Set Metrics Report")
    
    # display the KPI Cards
    metrics = st.session_state['eval_metrics']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Accuracy", metrics['Accuracy'])
    col2.metric("Macro Precision", metrics['Precision'])
    col3.metric("Macro Recall", metrics['Recall'])
    col4.metric("Macro F1-Score", metrics['F1-Score'])
    
    # display the confusion matrix
    st.markdown("### 🗺️ Confusion Matrix")
    import pandas as pd
    cm_df = pd.DataFrame(st.session_state['confusion_matrix'], index=class_names, columns=class_names)
    st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
    st.markdown("---")

# --- The main part for classifying uploaded images ---
st.markdown("## 📷 Single-Image Inference Test")
uploaded_file = st.file_uploader("Upload an apparel image to categorize (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col_img, col_pred = st.columns(2)
    
    with col_img:
        st.image(image, caption="Uploaded Fashion Image", use_container_width=True)
        
    with col_pred:
        if model is None:
            st.error("Weights file 'models/smart_catalog_resnet.pth' was not found. Please train the model first.")
        else:
            with st.spinner("Analyzing style patterns..."):
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    confidence, predicted_idx = torch.max(probabilities, 0)
                    
                predicted_class = class_names[predicted_idx.item()]
                conf_score = confidence.item() * 100
                
                st.success(f"### Prediction: **{predicted_class}**")
                st.metric("Model Confidence", f"{conf_score:.2f}%")
                
                st.markdown("#### Probability Distribution:")
                prob_data = {
                    "Apparel Class": class_names,
                    "Confidence Score (%)": [p.item() * 100 for p in probabilities]
                }
                import pandas as pd
                prob_df = pd.DataFrame(prob_data).sort_values(by="Confidence Score (%)", ascending=False)
                st.bar_chart(prob_df.set_index("Apparel Class"))