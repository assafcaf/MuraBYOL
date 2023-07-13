from MuraSimClr.denseNet.pipeline import get_study_level_data
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


study_types = ["XR_HAND", "XR_SHOULDER", "XR_ELBOW"]
base_dir = r"C:\Users\User\PycharmProjects\MURA-v1.1"


df_train = pd.DataFrame()
df_valid = pd.DataFrame()

for study_type in study_types:
    data = get_study_level_data(study_type, base_dir)
    df_train_per_study = data['train']
    df_train_per_study['study_type'] = study_type
    df_train_per_study['patient_num'] = df_train_per_study['Path'].apply(lambda x: x.split(os.sep)[-3].replace('patient', ''))
    df_train_per_study['study_num'] = df_train_per_study['Path'].apply(lambda x: x.split(os.sep)[-2].replace('study', ''))
    df_train = pd.concat([df_train, df_train_per_study])

    df_valid_per_study = data['valid']
    df_valid_per_study['study_type'] = study_type
    df_valid_per_study['patient_num'] = df_valid_per_study['Path'].apply(lambda x: x.split(os.sep)[-3].replace('patient', ''))
    df_valid_per_study['study_num'] = df_valid_per_study['Path'].apply(lambda x: x.split(os.sep)[-2].replace('study', ''))
    df_valid = pd.concat([df_valid, df_valid_per_study])

# num positive and negative images in each category

def num_positive_and_total_studies(df_train, df_valid):
    res_df = pd.DataFrame()
    for study_type in study_types:
        train = df_train[df_train['study_type'] == study_type]
        valid = df_valid[df_valid['study_type'] == study_type]
        train_pos = train[train['Label'] == 1]
        valid_pos = valid[valid['Label'] == 1]
        res_df = res_df.append({'study_type': study_type,
                                'train_positive': len(train_pos),
                                'train_total': len(train),
                                'valid_positive': len(valid_pos),
                                'valid_total': len(valid)}, ignore_index=True)

    res_df = res_df.set_index('study_type')
    sns.set(style='whitegrid')
    res_df.plot(kind='bar', stacked=True)
    plt.xlabel('Study Type')
    plt.ylabel('Count')
    plt.title('Study Type Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return res_df


# num count images for each study and study/category
def num_images_per_study(train_df, valid_df):
    res_df = pd.DataFrame()
    for study_type in study_types:
        train = train_df[train_df['study_type'] == study_type]
        valid = valid_df[valid_df['study_type'] == study_type]
        avg_count_train = train['Count'].mean()
        avg_count_valid = valid['Count'].mean()
        res_df = res_df.append({'study_type': study_type,
                                'train_avg_count': avg_count_train,
                                'valid_avg_count': avg_count_valid}, ignore_index=True)

    res_df = res_df.set_index('study_type')
    sns.set(style='whitegrid')
    res_df.plot(kind='bar', stacked=True)
    plt.xlabel('Study Type')
    plt.ylabel('Avg')
    plt.title('Average Number of Images per Study')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return res_df



def num_studies_per_patient(df_train, df_valid):
    for study_type in study_types:
        train = df_train[df_train['study_type'] == study_type]
        valid = df_valid[df_valid['study_type'] == study_type]

        for df in [train, valid]:

            studies_per_person = df.groupby('patient_num')['study_num'].nunique()

            sns.histplot(studies_per_person, bins=3)
            plt.xlabel('Number of Studies')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Studies per Patient for {study_type.capitalize()}')
            plt.show()


studies_distribution = num_positive_and_total_studies(df_train, df_valid)
num_images_per_study = num_images_per_study(df_train, df_valid)
num_studies_per_patient = num_studies_per_patient(df_train, df_valid)
