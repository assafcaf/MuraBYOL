import os
from sklearn.metrics import accuracy_score

import numpy as np
import torch
from torchvision import models
from dataset import build_ds
from tqdm import tqdm
from byol_pytorch.boyl_utils import NetWrapperFineTune
import time
from utils import fetch_data_for_fine_tuning


def train():
    # train model
    print("Training model...")
    ds_train = build_ds(train_pth, batch_size=batch_size, image_size=image_size)
    for epoch in range(n_epochs):
        losses = []
        time.sleep(0.05)
        for images, labels in tqdm(ds_train, desc=f"Training epoch {epoch + 1}", total=len(ds_train)):
            images, labels = images.to(device), labels.to(torch.float32).to(device)
            logits, _ = model(images)
            loss = criterion(logits.squeeze(1), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"Epoch {epoch + 1} loss: {np.mean(losses)}")


def evaluate():
    ds_valid = build_ds(valid_pth, batch_size=batch_size)
    print("Evaluate fine-tuned model...")
    time.sleep(0.05)
    y_true, y_pred = [], []
    for images, labels in tqdm(ds_valid, desc=f"Fine Tune evaluation...", total=len(ds_valid)):
        y_true.extend(list(labels.numpy()))
        images, labels = images.to(device), labels.to(torch.float32).to(device)
        logits, _ = model(images)
        y_pred.extend(list(logits.squeeze(1).detach().cpu().numpy() > 0.5))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"Fine Tune accuracy: {acc}")

def setup_model():
    resnet = models.resnet18()

    resnet.load_state_dict(torch.load(model_pth))

    model = NetWrapperFineTune(n_classes=1, net=resnet, projection_size=256, projection_hidden_size=4096,
                                layer="avgpool", use_simsiam_mlp=False)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    return model, opt, criterion


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")

    # Hyperparameters
    learning_rate = 1e-3
    image_size = 256
    batch_size = 128
    n_epochs = 1

    # Map to dataset folder location
    base_dire = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3"
    train_pth = os.path.join(base_dire, r"data\MiniMURA\train")
    valid_pth = os.path.join(base_dire, r"data\MiniMURA\valid")
    model_pth = os.path.join(base_dire, r"src\byol\trained_models\model1\improved-resnet18.pt")

    # setup model
    model, opt, criterion = setup_model()

    # train model
    train()

    # evaluate model
    evaluate()

    # Save model
    model_pth = os.path.join(os.path.dirname(os.path.realpath(model_pth)), "model-finetuned.pt")
    torch.save(model.state_dict(), model_pth)
