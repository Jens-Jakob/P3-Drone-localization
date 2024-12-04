# feature_extractor.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from transformers.models.superpoint.modeling_superpoint import simple_nms, top_k_keypoints


def resnet_feature_extractor():
    #Load ResNet50 model and use the updated 'weights' argument

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    #Remove the classification layer
    model = nn.Sequential(*list(model.children())[:-1])
    #model to evaluation mode
    model.eval()
    return model


# Function to load the saved model and remove the classification layer
def load_retrained_feature_extractor():
    model_path = "resnet50_model.pth"
    # Load the saved state dict
    checkpoint = torch.load(model_path)

    # Recreate the ResNet model and load state dict
    model = models.resnet50(weights=None)
    model.fc = nn.Identity()  # Remove the final fully connected layer
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the trained weights
    model.eval()  # Set to evaluation mode

    return model

def extract_features(model, img_tensor):
    #Extract features from the image using the model and flatten the output
    #This is only needed when training
    with torch.no_grad():
        features = model(img_tensor)
    #Flatten to 1D and convert to numpy(Cosine similarity takes 1D)
    return features.view(-1).numpy()




#https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html
def resnet152_feature_extractor():
    # Load ResNet152 model with pretrained weights
    model = models.resnet152(weights='IMAGENET1K_V1')

    # Remove the classification layer (the last fully connected layer)
    model.fc = nn.Identity()  # This replaces the classification layer with an identity layer

    # Create a list of layers to replace
    layers_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            layers_to_replace.append((name, nn.PReLU()))

    # Replace ReLU with PReLU after the iteration
    for name, new_layer in layers_to_replace:
        setattr(model, name, new_layer)

    model.eval()  # Set the model to evaluation mode
    return model



