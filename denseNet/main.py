
import torch
import os
from densenet import densenet169
from utils import plot_training, n_p, get_count
from train import train_model, get_metrics
from pipeline import get_study_level_data, get_full_data, get_dataloaders
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    # set data directory path
    # base_dir = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\MURA-v1.1"
    # case_study = 'XR_WRIST'
    # # #### load study level dict data
    # # study_data = get_study_level_data(study_type=case_study, base_dir=base_dir)
    # study_data = get_full_data(base_dir=base_dir)
    #
    # # #### Create dataloaders pipeline
    # data_cat = ['train', 'valid']  # data categories
    # dataloaders = get_dataloaders(study_data, batch_size=1, num_workers=1, study_level=False)
    # dataset_sizes = {x: len(study_data[x]) for x in data_cat}
    #
    #
    # # #### Build model
    # # tai = total abnormal images, tni = total normal images
    # tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    # tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    # Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}  # weight of positive class
    # Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}  # weight of negative class
    #
    #
    #
    # print('tai:', tai)
    # print('tni:', tni, '\n')
    # print('Wt0 train:', Wt0['train'])
    # print('Wt0 valid:', Wt0['valid'])
    # print('Wt1 train:', Wt1['train'])
    # print('Wt1 valid:', Wt1['valid'])

    ## TODO: ask rather we need the weights for each image or for each study
    class Loss(torch.nn.modules.Module):
        def __init__(self, Wt1, Wt0):
            super(Loss, self).__init__()
            self.Wt1 = Wt1
            self.Wt0 = Wt0

        def forward(self, inputs, targets, phase):
            loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
            return loss

    model = densenet169(pretrained=True)
    model = model.cuda()

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)


    # #### Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=15)

    torch.save(model.state_dict(), 'models/model.pth')

    get_metrics(model, criterion, dataloaders, dataset_sizes)


if __name__ == '__main__':
    main()

