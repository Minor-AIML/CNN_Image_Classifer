import io
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from vgg_model import VGGNet

app = Flask(__name__)

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load model checkpoint (update path if needed)
model_path = "models/best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGNet()
ckpt = torch.load(model_path, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
elif isinstance(ckpt, dict):
    model.load_state_dict(ckpt)
else:
    model.load_state_dict(ckpt)
model.to(device)
model.eval()

# Transform: Resize to 32x32, ToTensor, Normalize same as training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    try:
        img = Image.open(file.stream).convert('RGB')
    except:
        return jsonify({'error': 'Invalid image'}), 400

    # Preprocess
    img_t = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
    
    return jsonify({'prediction': class_name})

if __name__ == '__main__':
    app.run(debug=True)
