import numpy as np
import torch
from torchvision import models
from dataset import build_ds
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from byol_pytorch.boyl_utils import NetWrapper
import time


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")

    model_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\src\byol\trained_models\model1\improved-resnet18.pt"
    resnet_t = models.resnet18()
    resnet_r = models.resnet18()


    resnet_t.load_state_dict(torch.load(model_pth))
    resnet_t = NetWrapper(resnet_t, 256, 4096, layer="avgpool", use_simsiam_mlp=False)
    resnet_r = NetWrapper(resnet_r, 256, 4096, layer="avgpool", use_simsiam_mlp=False)

    resnet_r.to(device)
    resnet_t.to(device)

    # Map to dataset folder location
    train_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\train"
    image_size = 256
    batch_size = 256
    ds_train = build_ds(train_pth, batch_size=batch_size, image_size=image_size)

    x_r, x_t, y = [], [], []
    print("Generating representations for training linear model...")
    time.sleep(0.05)
    for images, labels in tqdm(ds_train, desc="Generate representation", total=len(ds_train)):
        with torch.no_grad():
            images = images.to(device)
            representations_r, _ = resnet_r(images)
            representations_t, _ = resnet_t(images)
            x_r.extend(list(representations_r.cpu().detach().numpy()))
            x_t.extend(list(representations_t.cpu().detach().numpy()))
            y.extend(list(labels.detach().numpy()))

    xr_train = np.array(x_r)
    xt_train = np.array(x_t)
    y_train = np.array(y)

    # train lenear regression model
    print("Training linear regression model...")
    reg_r = linear_model.LogisticRegression(max_iter=10000)
    reg_t = linear_model.LogisticRegression(max_iter=10000)
    reg_r.fit(xr_train, y_train)
    reg_t.fit(xt_train, y_train)

    # evaluate model
    x_r, x_t, y = [], [], []
    valid_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\valid"
    ds_valid = build_ds(valid_pth, batch_size=batch_size)
    print("Generating representations for testing linear model...")
    time.sleep(0.05)

    for images, labels in tqdm(ds_valid, desc=f"Generate representation", total=len(ds_valid)):
        with torch.no_grad():
            images = images.to(device)
            representations_t, _ = resnet_t(images)
            representations_r, _ = resnet_r(images)
            x_r.extend(list(representations_r.cpu().detach().numpy()))
            x_t.extend(list(representations_t.cpu().detach().numpy()))
            y.extend(list(labels.detach().numpy()))

    xr_test = np.array(x_r)
    xt_test = np.array(x_t)
    y_test = np.array(y)

    yr_pred = reg_r.predict(xr_test)
    yt_pred = reg_t.predict(xt_test)

    acc = accuracy_score(y_test, yr_pred)
    print(f"Accuracy for randomly resnet model: {acc:.3f}")

    acc = accuracy_score(y_test, yt_pred)
    print(f"Accuracy for trained resnet model: {acc:.3f}")

