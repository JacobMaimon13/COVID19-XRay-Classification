import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ChestXrayDataset
from src.models import SimpleCNN, get_resnet18
from src.utils import evaluate_model, plot_confusion_matrix
import pandas as pd

# CONFIG
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Running on {DEVICE}")
    
    # NOTE: This assumes you have already prepared the DataFrame 'train_df' and 'test_df'
    # Since the original code requires downloading 2GB+ of data, 
    # we recommend running the data preparation notebook first to save the CSVs.
    try:
        train_df = pd.read_csv('data/train.csv')
        test_df = pd.read_csv('data/test.csv')
    except FileNotFoundError:
        print("Error: data CSVs not found. Please prepare data first.")
        return

    # Datasets
    train_ds = ChestXrayDataset(train_df, augment_covid=True)
    test_ds = ChestXrayDataset(test_df)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model Selection (Change to SimpleCNN or get_resnet18) ---
    model = get_resnet18(num_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
        
    # Evaluation
    print("Evaluating...")
    acc, labels, preds = evaluate_model(model, test_loader, DEVICE)
    print(f"Test Accuracy: {acc:.4f}")
    plot_confusion_matrix(labels, preds)

if __name__ == "__main__":
    main()
