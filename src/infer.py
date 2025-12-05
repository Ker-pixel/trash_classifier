import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import build_model

MODEL_PATH = "../models/resnet50_binary.pth"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    # Confidence fallback
    if conf.item() < 0.55:
        return "not recyclable (no)"

    return "recyclable (yes)" if pred.item() == 1 else "not recyclable (no)"

if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))