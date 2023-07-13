import numpy as np
import torch
from torchvision import models
from dataset import build_ds
from tqdm import tqdm

from sklearn import linear_model
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")

    model_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\src\byol\trained_models\model3\improved-resnet18.pt"
    resnet_t = models.resnet18()
    resnet_t.load_state_dict(torch.load(model_pth))
    resnet_r = models.resnet18()

    resnet_r.to(device)
    resnet_t.to(device)

    # Map to dataset folder location
    train_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MiniMURA\train"
    ds_train = build_ds(train_pth, batch_size=64)

    x_r, x_t, y = [], [], []
    for images, labels in tqdm(ds_train, desc=f"Testing model", total=len(ds_train)):
        images = images.to(device)
        representations_r = resnet_r(images)
        representations_t = resnet_t(images)
        x_r.extend(list(representations_r.cpu().detach().numpy()))
        x_t.extend(list(representations_t.cpu().detach().numpy()))
        y.extend(list(labels.detach().numpy()))

    xr_train = np.array(x_r)
    xt_train = np.array(x_t)
    y_train = np.array(y)

    # train lenear regression model
    print("Training linear regression model...")
    reg_r = linear_model.LogisticRegression(max_iter=1000)
    reg_t = linear_model.LogisticRegression(max_iter=1000)
    reg_r.fit(xr_train, y_train)
    reg_t.fit(xt_train, y_train)

    # evaluate model
    x_r, x_t, y = [], [], []
    valid_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MiniMURA\valid"
    ds_valid = build_ds(train_pth, batch_size=64)
    for images, labels in tqdm(ds_train, desc=f"Testing model", total=len(ds_valid)):
        images = images.to(device)
        representations_t = resnet_t(images)
        representations_r = resnet_r(images)
        x_r.extend(list(representations_r.cpu().detach().numpy()))
        x_t.extend(list(representations_t.cpu().detach().numpy()))
        y.extend(list(labels.detach().numpy()))

    xr_test = np.array(x_r)
    xt_test = np.array(x_t)
    y_test = np.array(y)

    yr_pred = reg_r.predict(xr_test)
    yt_pred = reg_t.predict(xt_test)

    acc = accuracy_score(y_test, yr_pred)
    print(f"Accuracy r: {acc:.3f}")

    acc = accuracy_score(y_test, yt_pred)
    print(f"Accuracy t: {acc:.3f}")

