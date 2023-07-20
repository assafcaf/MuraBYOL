import os
import time
import copy
import pandas as pd
import torch
import argparse
from torch.autograd import Variable
from denseNet.densenet import densenet169
from denseNet.utils import plot_training, n_p, get_count
from denseNet.train import train_model, get_metrics
from denseNet.pipeline import get_study_level_data, get_dataloaders


def arg_parse():
    parser = argparse.ArgumentParser(description='MURA data organization')
    parser.add_argument('--data-in', help='input data location', default=r"data\processed_100",
                        type=str)
    parser.add_argument('--batch-size', help='batch size for training', default=8,
                        type=int)
    parser.add_argument('--num-epochs', help='number of epochs', default=5,
                        type=int)
    parser.add_argument('--learning-rate', help='learning rate', default=0.0001,
                        type=float)
    parser.add_argument('--model-dir', help='location to store the trained model',
                        default=r"trained_models\densnet", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    # #### load study level dict data
    for study_type in ['XR_WRIST', 'XR_ELBOW', 'XR_HAND', 'XR_SHOULDER']:
        study_data = get_study_level_data(study_type=study_type, data_dir=args.data_in)

        # #### Create dataloaders pipeline
        data_cat = ['train', 'valid']  # data categories
        dataloaders = get_dataloaders(study_data, batch_size=args.batch_size)
        dataset_sizes = {x: len(study_data[x]) for x in data_cat}

        # #### Build model
        # tai = total abnormal images, tni = total normal images
        tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
        tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
        Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
        Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

        print('tai:', tai)
        print('tni:', tni, '\n')
        print('Wt0 train:', Wt0['train'])
        print('Wt0 valid:', Wt0['valid'])
        print('Wt1 train:', Wt1['train'])
        print('Wt1 valid:', Wt1['valid'])


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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

        # #### Train model
        model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=args.num_epochs)
        torch.save(model.state_dict(), os.path.join(args.model_dir, f"densnet_{study_type}.pth"))

        get_metrics(model, criterion, dataloaders, dataset_sizes)