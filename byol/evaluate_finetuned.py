import os
import torch
from torchvision import models
from byol_pytorch.boyl_utils import NetWrapperFineTune
from dataset import build_ds
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_dire = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3"
    model_pth = os.path.join(base_dire, r"src\byol\trained_models\model1\model-finetuned.pt")


    model = NetWrapperFineTune(n_classes=1, net=models.resnet18(), projection_size=256, projection_hidden_size=4096,
                               layer="avgpool", use_simsiam_mlp=False)
    model.load_state_dict(torch.load(model_pth), strict=False)
    model.to(device)

    batch_size = 128
    valid_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\valid"
    ds_valid = build_ds(valid_pth, batch_size=batch_size, rfn=False)
    print("Generating representations for testing linear model...")

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