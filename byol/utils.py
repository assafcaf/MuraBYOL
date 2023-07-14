import torch
import os
import json
import math
import random
from sklearn.metrics import accuracy_score
import pandas as pd
import shutil

def validate(ds, model, device='cuda'):
    losses = []
    with torch.no_grad():
        for images, labels in ds:
            images = images.to(device)
            loss = model(images)  # update moving average of target encoder
            losses.append(loss.item())
    return sum(losses)/len(losses)


def save_model(model, path, model_name, meta_data=None):
    model_dir = os.path.join(path, "model" + str(len(os.listdir(path))+1))
    # create model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save resnet model
    torch.save(model.state_dict(), os.path.join(model_dir, f'improved-{model_name}.pt'))

    # save meta data about learning process
    json_object = json.dumps(meta_data, indent=4)
    # Writing to sample.json
    with open(os.path.join(model_dir, "training_metadata.json"), "w") as outfile:
        outfile.write(json_object)

def fetch_data_for_fine_tuning(p, verify_study_type_balance=False,
                               verify_patient_balance=False):
    input_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed"
    output_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\MURAFineTuning"

    # Create the MURA v2 root folder
    os.makedirs(output_pth, exist_ok=True)
    train_path = os.path.join(input_pth, 'train')
    labels = os.listdir(train_path)

    for label in labels:
        label_path = os.path.join(train_path, label)
        files = os.listdir(label_path)
        if verify_study_type_balance:
            # Since there aren't any "returning patients" for different study types, we can just sample from the study types
            study_types = set([f"{f.split('_')[0]}_{f.split('_')[1]}" for f in files])
            for study_type in study_types:
                study_type_files = [f for f in files if study_type in f]
                sample_size = math.floor(len(study_type_files) * p)
                sample_images = random.sample(study_type_files, sample_size)
                for image in sample_images:
                    shutil.copy2(os.path.join(label_path, image), os.path.join(output_pth, image))
                print(
                    f"Data for class {label} and study type {study_type} was fetched for fine tuning. {len(sample_images)} images were sampled")
        else:
            sample_size = int(len(files) * p)
            sample_images = random.sample(files, sample_size)
            for image in sample_images:
                shutil.copy2(os.path.join(label_path, image), os.path.join(output_pth, image))
            print(
                f"Data for class {label} was fetched for fine tuning. {len(sample_images)} images were sampled")


def calculate_accuracy_per_study(x_names, y_pred):
    """
    Calculate the accuracy per study
    :param x_names: The names of the images
    :param y_pred: The predicted labels per image
    :return: The accuracy per study
    """
    # Create a dataframe with the predictions and the ground truth
    df = pd.DataFrame({"file_names": x_names, "y_pred": y_pred})
    # Group the dataframe by the study
    df['y_true'] = df.file_names.apply(lambda x: 1 if x.split("_")[5] == 'positive' else 0)
    df['study'] = df.file_names.apply(lambda x: x.split("_")[2] + "_" + x.split("_")[3])
    per_study_gb = df.groupby('study')
    per_study_pred = per_study_gb['y_pred'].mean()
    y_true = per_study_gb['y_true'].max()
    per_study_pred['final_pred'] = per_study_pred['y_pred'].apply(lambda x: 1 if x > 0.5 else 0)
    # Calculate the accuracy per study
    per_study_acc = accuracy_score(y_true, per_study_pred['final_pred'])
    return per_study_acc