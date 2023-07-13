import torch
import os
import json
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
