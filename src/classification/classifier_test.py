import cv2
import numpy as np
import torch
import torchvision


def predict_class(image_path, weights_path, device="auto"):

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = torchvision.models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, 3)
    )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    image = cv2.imread(str(image_path), dtype=np.uint8)

    if image is None:
        raise ValueError(f"Failed to read the file: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    tensor = (tensor - mean) / std

    with torch.no_grad():
        output = model(tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return predicted_class
