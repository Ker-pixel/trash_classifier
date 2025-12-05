from flask import Flask, request, render_template_string
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.model import build_model

MODEL_PATH = "models/resnet50_binary.pth"

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Recycling Classifier</title>
</head>
<body style="font-family: Arial; padding: 40px;">
<h1>♻️ Recycling Classifier</h1>
<p>Upload an image:</p>

<form action="/predict" method="POST" enctype="multipart/form-data">
  <input type="file" name="file">
  <button type="submit">Analyze</button>
</form>

{% if result %}
  <h2>Result: {{ result }}</h2>
{% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    device = torch.device("cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    if conf.item() < 0.55:
        return render_template_string(HTML, result="not recyclable (no)")

    res = "recyclable (yes)" if pred.item() == 1 else "not recyclable (no)"
    return render_template_string(HTML, result=res)


if __name__ == "__main__":
    app.run()