import torch
from byol_pytorch import BYOL
from torchvision import models
from dataset import build_ds
from tqdm import tqdm
import time
from utils import validate, save_model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device '{device}'.")
    model_name = "resnet18"

    # training parameters
    batch_size = 64
    image_size = 224
    num_epochs = 25
    num_workers = 4
    learning_rate = 7.5e-5
    hidden_layer = 'avgpool'
    debug = False
    resnet = eval(f"models.{model_name}(pretrained=True)").to(device)
    learner = BYOL(
        resnet,
        image_size=image_size,
        hidden_layer=hidden_layer
    )
    opt = torch.optim.Adam(learner.parameters(), lr=learning_rate)


    # Map to dataset folder location
    train_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\train"
    valid_pth = r"C:\studies\IDC_dataScience\year_B\AIForHealthcare\HW3\data\MURAProcessed\valid"

    ds_train = build_ds(train_pth, batch_size=batch_size, image_size=image_size)
    ds_valid = build_ds(valid_pth, batch_size=batch_size,  image_size=image_size)


    valid_loss = validate(ds_valid, learner)
    print(f"initial validation loss: {valid_loss:.3f}")

    # training loop
    for epoch in range(num_epochs):
        losses = []
        time.sleep(0.05)
        for images, labels in tqdm(ds_train, desc=f"Training epoch {epoch+1}", total=len(ds_train)):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # update moving average of target encoder
            losses.append(loss.item())
        valid_loss = validate(ds_valid, learner)
        print(f"epoch {epoch+1}/{num_epochs}, loss={sum(losses)/len(losses):.3f}, "
              f"validation_loss={valid_loss:.3f}")

    meta_data = {"epochs": num_epochs,
                 "batch_size": batch_size,
                 "image_size": image_size,
                 "hidden_layer": hidden_layer,
                 "optimizer": "Adam",
                 "learning_rate": learning_rate,
                 "device": device,
                 "model_name": model_name,
                 'valid_loss': valid_loss}

    # save your improved network
    if not debug:
        save_model(resnet, "trained_models", model_name, meta_data=meta_data)
        print("model saved")
