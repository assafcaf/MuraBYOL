import os
import time
import torch
from torchvision import models
from byol.dataset import build_ds
from byol.model.boyl import NetWrapperFineTune
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils import calculate_accuracy_per_study
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default=r"data\processed_100",
                        type=str)
    parser.add_argument('--batch-size', help='batch size for training', default=64,
                        type=int)
    parser.add_argument('--image-size', help='image size', default=224,
                        type=int)
    parser.add_argument('--model-pth', help='output data location',
                        default=r"trained_models\boyl\model1\model-finetuned-processed_100.pt", type=str)
    parser.add_argument('--threshold', help='output data location',
                        default=0, type=float)
    parser.add_argument('--split2bones', help='output data location', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    bone_names = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")

    args = arg_parse()
    model = NetWrapperFineTune(n_classes=1, net=models.resnet18(), projection_size=256, projection_hidden_size=4096,
                               layer="avgpool", use_simsiam_mlp=False)
    model.load_state_dict(torch.load(args.model_pth), strict=False)
    model.to(device)
    model.eval()

    ds_valid = build_ds(os.path.join(args.data_in, "valid"), batch_size=args.batch_size, rfn=True, shuffle=False)
    print("Generating representations for testing linear model...")
    print("Evaluate fine-tuned model...")
    time.sleep(0.05)
    y_true, y_pred, f_names = [], [], []
    for images, labels, f_name in tqdm(ds_valid, desc=f"Fine Tune evaluation...", bar_format='{l_bar}{bar:10}{r_bar}'):
        y_true.extend(list(labels.numpy()))
        images = images.to(device)
        logits = model(images)
        y_pred.extend(list(F.sigmoid(logits).squeeze(1).detach().cpu().numpy() > 0.5))
        f_names.extend(f_name)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f_names = np.array(f_names)
    if args.split2bones:
        for bone in bone_names:
            idx = np.array([f_name.split("/")[-1].split("_")[1] == bone for f_name in f_names])
            metrics = calculate_accuracy_per_study(f_names[idx], y_pred[idx], threshold=args.threshold)
            print(f"|{bone:>10}|: accuracy: {metrics['acc']:.3f}, precision: {metrics['precision']:.3f},"
                  f" recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")
    else:
        metrics = calculate_accuracy_per_study(f_names, y_pred, threshold=args.threshold)
        print(f"Fine Tune accuracy: {metrics['acc']:.3f}, precision: {metrics['precision']:.3f},"
              f" recall: {metrics['recall']:.3f}, f1: {metrics['f1']:.3f}")