import torch
import os
import json
import math
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    os.makedirs(path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # save resnet model
    torch.save(model.state_dict(), os.path.join(model_dir, f'improved-{model_name}.pt'))

    # save meta data about learning process
    json_object = json.dumps(meta_data, indent=4)
    # Writing to sample.json
    with open(os.path.join(model_dir, "training_metadata.json"), "w") as outfile:
        outfile.write(json_object)

def fetch_data_for_fine_tuning(p, verify_study_type_balance=False, input_pth="", output_pth=""):
    # Create the MURA v2 root folder

    os.makedirs(output_pth, exist_ok=True)
    output_pth = os.path.join(output_pth, "train")
    os.makedirs(output_pth, exist_ok=True)
    train_path = os.path.join(input_pth, 'train')
    labels = os.listdir(train_path)

    for label in labels:
        label_path = os.path.join(train_path, label)
        os.makedirs(os.path.join(output_pth, label), exist_ok=True)
        files = os.listdir(label_path)
        if verify_study_type_balance:
            # Since there aren't any "returning patients" for different study types, we can just sample from the study types
            study_types = set([f"{f.split('_')[0]}_{f.split('_')[1]}" for f in files])
            for study_type in study_types:
                study_type_files = [f for f in files if study_type in f]
                sample_size = math.floor(len(study_type_files) * p)
                sample_images = random.sample(study_type_files, sample_size)
                for image in sample_images:
                    shutil.copy2(os.path.join(label_path, image), os.path.join(output_pth, label, image))
        else:
            sample_size = int(len(files) * p)
            sample_images = random.sample(files, sample_size)
            for image in sample_images:
                shutil.copy2(os.path.join(label_path, image), os.path.join(output_pth, image))
            print(
                f"Data for class {label} was fetched for fine tuning. {len(sample_images)} images were sampled")


def calculate_accuracy_per_study(x_names, y_pred, threshold=0.1):
    """
    Calculate the accuracy per study
    :param x_names: The names of the images
    :param y_pred: The predicted labels per image
    :return: The accuracy per study
    """
    # Create a dataframe with the predictions and the ground truth
    df = pd.DataFrame({"file_names": x_names, "y_pred": y_pred})
    # Group the dataframe by the study
    df['y_true'] = df.file_names.apply(lambda x: 1 if x.split("_")[4] == 'positive' else 0)
    df['study'] = df.file_names.apply(lambda x: x.split("_")[2] + "_" + x.split("_")[3])
    per_study_gb = df.groupby('study')
    per_study_pred = per_study_gb['y_pred'].mean()
    y_true = per_study_gb['y_true'].max()
    per_study_pred['final_pred'] = per_study_pred.apply(lambda x: 1 if x > threshold else 0)

    # Calculate the accuracy per study
    acc = accuracy_score(y_true, per_study_pred['final_pred'])
    precision = precision_score(y_true, per_study_pred['final_pred'])
    recall = recall_score(y_true, per_study_pred['final_pred'])
    f1 = f1_score(y_true, per_study_pred['final_pred'])

    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def organize_data(input_data_location, output_data_location):
    # Create the MURA v2 root folder
    mura_v2_folder = output_data_location
    os.makedirs(mura_v2_folder, exist_ok=True)

    # Iterate through the input data hierarchy
    for split in [d for d in os.listdir(input_data_location) if os.path.isdir(os.path.join(input_data_location, d))]:
        split_folder = os.path.join(input_data_location, split)
        output_folder = os.path.join(mura_v2_folder, split)
        os.makedirs(output_folder, exist_ok=True)
        for study_type in os.listdir(split_folder):
            study_type_folder = os.path.join(split_folder, study_type)
            for patient in os.listdir(study_type_folder):
                patient_folder = os.path.join(study_type_folder, patient)
                for study in os.listdir(patient_folder):
                    study_folder = os.path.join(patient_folder, study)
                    for image in os.listdir(study_folder):

                        # Ignore corrupted images
                        if image[0] == ".":
                            print(f"Skipping {os.path.join(study_folder, image)}")
                            continue
                        # Determine the label based on the original file path
                        if "positive" in study:
                            label_folder = os.path.join(output_folder, "positive")
                        else:
                            label_folder = os.path.join(output_folder, "negative")

                        os.makedirs(label_folder, exist_ok=True)  # Create the label folder if it doesn't exist

                        new_filename = f"{study_type}_{patient}_{study}_{image}"
                        # Copy the file to the label folder
                        shutil.copy2(os.path.join(study_folder, image), os.path.join(label_folder, new_filename))

    # Print a message when the data organization is complete
    print("Data organization complete!")

