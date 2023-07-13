from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class MyDataClass(Dataset):
    def __init__(self, image_path, transform=None):
        super(MyDataClass, self).__init__()
        self.data = datasets.ImageFolder(image_path,  transform)    # Create data from folder

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

    def __len__(self):
        return len(self.data)

#TODO: validate build ds
def build_ds(image_path, transform=None, batch_size=32, shuffle=True, num_workers=1,
             pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True,
             collate_fn=None, sampler=None, timeout=0, worker_init_fn=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])
    data_class = MyDataClass(image_path, transform)
    data_loader = DataLoader(data_class, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=drop_last, prefetch_factor=prefetch_factor,
                             persistent_workers=persistent_workers, collate_fn=collate_fn, sampler=sampler,
                             timeout=timeout, worker_init_fn=worker_init_fn)
    return data_loader


