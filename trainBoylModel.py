import os.path

import torch
from byol.model.boyl import BYOL
from torchvision import models
from byol.dataset import build_ds
from tqdm import tqdm
import time
from utils import validate, save_model
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default=r"data\processed_100", type=str)
    parser.add_argument('--batch-size', help='input data location', default=84, type=int)
    parser.add_argument('--image-size', help='input data location', default=224, type=int)
    parser.add_argument('--epochs', help='input data location', default=20, type=int)
    parser.add_argument('--learning-rate', help='input data location', default=7e-4, type=float)
    parser.add_argument('--model-name', help='output data location', default="resnet18", type=str)
    parser.add_argument('--moving-average-decay', help='output data location', default=0.99, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")

    hidden_layer = 'avgpool'
    resnet = eval(f"models.{args.model_name}(pretrained=True)").to(device)
    learner = BYOL(
        resnet,
        image_size=args.image_size,
        hidden_layer=hidden_layer,
        moving_average_decay=args.moving_average_decay
    )
    opt = torch.optim.Adam(learner.parameters(), lr=args.learning_rate)
    learner = learner.to(device)

    # load data
    ds_train = build_ds(os.path.join(args.data_in, "train"), batch_size=args.batch_size, image_size=args.image_size)
    ds_valid = build_ds(os.path.join(args.data_in, "valid"), batch_size=args.batch_size, image_size=args.image_size)


    valid_loss = validate(ds_valid, learner)
    print(f"initial validation loss: {valid_loss:.3f}")

    # training loop
    for epoch in range(args.epochs):
        losses = []
        time.sleep(0.05)
        for images, labels in tqdm(ds_train, desc=f"Training epoch {epoch+1}", total=len(ds_train),
                                   bar_format='{l_bar}{bar:10}{r_bar}'):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of target encoder
            losses.append(loss.item())
        valid_loss = validate(ds_valid, learner)
        print(f"epoch {epoch+1}/{args.epochs}, loss={sum(losses)/len(losses):.3f}, "
              f"validation_loss={valid_loss:.3f}")

    meta_data = {"epochs": args.epochs,
                 "batch_size": args.batch_size,
                 "image_size": args.image_size,
                 "hidden_layer": hidden_layer,
                 "optimizer": "Adam",
                 "learning_rate": args.learning_rate,
                 "device": device,
                 "model_name": args.model_name,
                 'valid_loss': valid_loss,
                 'moving_average_decay': args.moving_average_decay}

    # save your improved network
    save_model(resnet, os.path.join("trained_models", "boyl"), args.model_name, meta_data=meta_data)
    print("model saved")
