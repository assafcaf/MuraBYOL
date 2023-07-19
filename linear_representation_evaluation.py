import os
import numpy as np
import torch
from torchvision import models
from byol.dataset import build_ds
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from byol.model.boyl import NetWrapper
from utils import calculate_accuracy_per_study
import time
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default=r"data\processed_100",
                        type=str)
    parser.add_argument('--batch-size', help='input data location', default=64,
                        type=int)
    parser.add_argument('--image-size', help='input data location', default=224,
                        type=int)
    parser.add_argument('--threshold', help='output data location',
                        default=0, type=float)
    parser.add_argument('--model-pth', help='output data location',
                        default=r"trained_models\boyl\model1\improved-resnet18.pt", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")
    args = arg_parse()
    resnet_t = models.resnet18()
    resnet_r = models.resnet18()

    resnet_t.load_state_dict(torch.load(args.model_pth))
    resnet_t = NetWrapper(resnet_t, 256, 4096, layer="avgpool", use_simsiam_mlp=False)
    resnet_r = NetWrapper(resnet_r, 256, 4096, layer="avgpool", use_simsiam_mlp=False)

    resnet_r.to(device)
    resnet_t.to(device)

    # Map to dataset folder location
    ds_train = build_ds(os.path.join(args.data_in, "train"), batch_size=args.batch_size, image_size=args.image_size)

    # generate representations
    x_r, x_t, y = [], [], []
    print("Generating representations for training linear model...")
    time.sleep(0.05)
    for images, labels in tqdm(ds_train, desc="Generate representation", total=len(ds_train),
                               bar_format='{l_bar}{bar:10}{r_bar}'):
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
    x_r, x_t, f_names = [], [], []
    ds_valid = build_ds(os.path.join(args.data_in, "valid"), batch_size=args.batch_size, rfn=True)
    print("Generating representations for testing linear model...")
    time.sleep(0.05)

    for images, labels, f_name in tqdm(ds_valid, desc=f"Fine Tune evaluation...", bar_format='{l_bar}{bar:10}{r_bar}'):
        with torch.no_grad():
            images = images.to(device)
            representations_t, _ = resnet_t(images)
            representations_r, _ = resnet_r(images)
            x_r.extend(list(representations_r.cpu().detach().numpy()))
            x_t.extend(list(representations_t.cpu().detach().numpy()))
            f_names.extend(f_name)

    xr_test = np.array(x_r)
    xt_test = np.array(x_t)
    y_test = np.array(y)

    yr_pred = reg_r.predict(xr_test)
    yt_pred = reg_t.predict(xt_test)

    metrics = calculate_accuracy_per_study(f_names, yr_pred, threshold=args.threshold)
    print(f"Untrained ResNet representation: {metrics['acc']:.3f}, precision: {metrics['precision']:.3f},"
          f" recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")

    metrics = calculate_accuracy_per_study(f_names, yt_pred, threshold=args.threshold)
    print(f"trained ResNet representation: {metrics['acc']:.3f}, precision: {metrics['precision']:.3f},"
          f" recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")

