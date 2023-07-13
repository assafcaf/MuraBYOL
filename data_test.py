import os
import shutil
from tqdm import tqdm

def organize_data(input_data_location, output_data_location):
    # Create the MURA v2 root folder
    mura_v2_folder = output_data_location
    os.makedirs(mura_v2_folder, exist_ok=True)

    # Iterate through the input data hierarchy
    for split in [d for d in os.listdir(input_data_location) if os.path.isdir(os.path.join(input_data_location,d))]:
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
                        # Determine the label based on the original file path
                        if "positive" in study:
                            label_folder = os.path.join(output_folder, "positive")
                        else:
                            label_folder = os.path.join(output_folder, "negative")

                        os.makedirs(label_folder, exist_ok=True)  # Create the label folder if it doesn't exist

                        new_filename = f"{study_type}_{patient}_{study}_{image}"
                        # Copy the file to the label folder
                        if not image.startswith("."):
                            shutil.copy2(os.path.join(study_folder, image), os.path.join(label_folder, new_filename))
            break
    # Print a message when the data organization is complete
    print("Data organization complete!")

# Specify the input and output data locations
input_data_location = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURA-v1.1"
output_data_location = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MiniMURA"

# Call the function to organize the data
organize_data(input_data_location, output_data_location)