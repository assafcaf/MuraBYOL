import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import torch
from torchvision import models
from byol.dataset import build_ds
from tqdm import tqdm
from byol.model.boyl import NetWrapperFineTune
import time
import torch.nn.functional as F
import argparse
from utils import calculate_accuracy_per_study


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default=r"data\processed_100",
                        type=str)
    parser.add_argument('--batch-size', help='batch size for training', default=64,
                        type=int)
    parser.add_argument('--image-size', help='image size', default=224,
                        type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=20,
                        type=int)
    parser.add_argument('--num-workers', help='number of workers for data loader', default=4,
                        type=int)
    parser.add_argument('--learning-rate', help='learning rate', default=5e-4,
                        type=float)
    parser.add_argument('--positive-weights', help='weight of positive samples in the loss', default=2,
                        type=float)
    parser.add_argument('--model-dir', help='location to store the trained model',
                        default=r"trained_models\boyl\model1", type=str)
    parser.add_argument('--model-name', help='output data location',
                        default="resnet18", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")
    args = arg_parse()

    # data setup
    ds_type = args.data_in.split("\\")[-1]
    train_pth = os.path.join(args.data_in, "train")
    model_pth = os.path.join(args.model_dir, "improved-resnet18.pt")
    ds_train = build_ds(train_pth, batch_size=args.batch_size, image_size=args.image_size)
    positive_freq = sum(ds_train.dataset.data.targets) / len(ds_train.dataset.data.targets)

    # setup model
    resnet = models.resnet18()
    resnet.load_state_dict(torch.load(model_pth), strict=False)
    model = NetWrapperFineTune(n_classes=1, net=resnet, projection_size=256, projection_hidden_size=4096,
                                layer="avgpool", use_simsiam_mlp=False)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_freq)*args.positive_weights)

    # train model
    print("Training model...")
    for epoch in range(args.num_epochs):
        losses = []
        time.sleep(0.05)
        for images, labels in tqdm(ds_train, desc=f"Training epoch: {epoch + 1}", total=len(ds_train),
                                   bar_format='{l_bar}{bar:10}{r_bar}'):
            images, labels = images.to(device), labels.to(torch.float32).to(device)
            logits = model(images)
            loss = criterion(logits.squeeze(1), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch + 1} loss: {np.mean(losses):.3f}")

    # Save model
    model_pth = os.path.join(os.path.dirname(os.path.realpath(model_pth)), f"model-finetuned-{ds_type}.pt")
    torch.save(model.state_dict(), model_pth)
