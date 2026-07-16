import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Solving Python internal path issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset import get_fashion_loaders
from model_arch import CustomResNet

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1.The batch_size has been modified to 16 to be lightweight and very fast on the CPU
    train_loader, val_loader, test_loader, classes = get_fashion_loaders(batch_size=16)
    
    # 2.Initializing the model from scratch
    model = CustomResNet(num_classes=len(classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 2  # Reducing epochs to 2 to speed up extracting metrics immediately for discussion
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    
    print("--- Starting Training Loop ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Adding a step counter to monitor the movement in the Terminal
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            # Printing live every 100 steps to make sure the model is training successfully
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{step}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")
            
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"==> Finished Epoch {epoch+1}/{epochs} | Val Accuracy: {val_acc*100:.2f}%")
        
        # saving the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes
            }, "models/smart_catalog_resnet.pth")
            
    print("Training finished successfully. Evaluating on Test Set...")
    
    # 3. Final evaluation and calculating the confusion matrix and required metrics
    checkpoint = torch.load("models/smart_catalog_resnet.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())
            
    # calculating the required metrics
    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted')
    cm = confusion_matrix(test_labels, test_preds)
    
    print("\n========= FINAL EVALUATION METRICS =========")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("============================================")

if __name__ == "__main__":
    train_and_evaluate()