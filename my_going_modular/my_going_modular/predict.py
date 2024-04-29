
from pathlib import Path
import torch

import model_builder
# from my_going_modular import model_builder
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

device = "gpu" if torch.cuda.is_available() else "cpu"


def load_model(model_path_str: str, device: str) -> torch.nn.Module:

    model_path = Path(model_path_str)
    model = model_builder.get_default_tiynvgg(device=device)

    model.load_state_dict(torch.load(model_path))

    return model


def pred_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list[str] = None,
    transform=None,
    device: torch.device = device,
):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Print prediction and prediction probability

    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    print(title)



import argparse

# Creating a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument("--image", help="target image filepath to predict on")

# Get a model path
parser.add_argument(
    "--model_path",
    default="models/05_going_modular_script_mode_tinyvgg_model.pth",
    type=str,
    help="target model to use for prediction filepath",
)
args = parser.parse_args()


class_names = ["pizza", "steak", "sushi"]
im_path = Path("data/pizza_steak_sushi/test/pizza/1152100.jpg")
model_path_str = "models/05_going_modular_script_mode_tinyvgg_model.pth"



# laod model createor of model from dict

model = load_model(model_path_str=model_path_str, device=device)
data_transform = transforms.Compose([transforms.Resize((64, 64))])

# get predictions

pred_image(
    model=model,
    image_path=im_path,
    class_names=class_names,
    transform=data_transform,
    device=device,
)
