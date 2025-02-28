import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load an image and preprocess it
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Function to get the class label
def get_class_label(preds):
    _, class_index = torch.max(preds, 1)
    return class_index.item()

def get_conv_layer(model, conv_layer_name):
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Layer '{conv_layer_name}' not found in the model.")

# Function to generate Grad-CAM heatmap
def compute_gradcam(model, img_tensor, class_index, conv_layer_name="features.18"):
    conv_layer = get_conv_layer(model, conv_layer_name)

    # Forward hook to store activations
    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    hook = conv_layer.register_forward_hook(forward_hook)

    # Compute gradients
    img_tensor.requires_grad_(True)
    preds = model(img_tensor)
    loss = preds[:, class_index]
    model.zero_grad()
    loss.backward()

    # Get gradients
    grads = img_tensor.grad.cpu().numpy()
    pooled_grads = np.mean(grads, axis=(0, 2, 3))

    # Remove the hook
    hook.remove()

    activations = activations.detach().cpu().numpy()[0]
    for i in range(pooled_grads.shape[0]):
        activations[i, ...] *= pooled_grads[i]

    heatmap = np.mean(activations, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Overlay heatmap on image
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

if __name__ == "__main__":
    # Load a pretrained model (MobileNetV2)
    model = models.mobilenet_v2(pretrained=True)
    model.eval()

    # Example Usage
    img_path = "images/dog-2.jpg"  # Replace with your image path
    img_tensor = preprocess_image(img_path)

    # Get model predictions
    with torch.no_grad():
        preds = model(img_tensor)
    class_index = get_class_label(preds)

    print(f"Predicted Class Index: {class_index}")

    # Compute Grad-CAM heatmap
    heatmap = compute_gradcam(model, img_tensor, class_index)

    # Overlay heatmap on the original image
    output_img = overlay_heatmap(img_path, heatmap)

    # Save the heatmap
    cv2.imwrite("heatmap/2.jpg", output_img)
