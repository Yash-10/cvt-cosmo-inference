import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAMRegressor:
    def __init__(self, model, target_layer, ground_truth_param_value, index):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None  # Placeholder for feature maps
        self.gradients = None  # Placeholder for gradients
        self.ground_truth_param_value = ground_truth_param_value
        self.index = index

        # Set the model to evaluation mode
        self.model.eval()

        # Register a hook to capture gradients of the target layer
        self.hook = self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks for both forward and backward passes
        forward_hook_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_hook_handle = self.target_layer.register_full_backward_hook(backward_hook)

        return forward_hook_handle, backward_hook_handle

    def remove_hooks(self):
        # Remove hooks after usage
        self.hook[0].remove()
        self.hook[1].remove()

    def generate_gradcam(self, input_tensor):
        # Forward pass
        input_tensor.requires_grad_()
        self.model.zero_grad()
        model_output = self.model(input_tensor)[:, self.index]

        D = model_output - self.ground_truth_param_value
        # d = 1 / D

        # Backward pass to compute gradients
        model_output.backward(torch.ones_like(model_output), retain_graph=True)

        # The approach is from https://arxiv.org/pdf/2304.08192.pdf
        self.gradients *= -1 / (D**2)

        # Retrieve gradients and feature maps
        gradients = self.gradients  # Gradients of the target layer
        feature_maps = self.feature_maps  # Output of the target layer
        
        print(f'Gradients shape: {gradients.shape}')
        print(f'Feature maps shape: {feature_maps.shape}')

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # weight the channels by corresponding gradients
        for i in range(feature_maps.size()[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations and squeeze across the batch dimension since we assume only one image is passed.
        # Note: If want to support multiple images at once, need to remove squeeze(dim=0) and instead add mean(dim=0) to average cams across batch images.
        cam = torch.mean(feature_maps, dim=1).squeeze(dim=0)

        # relu on top of the heatmap
        cam = F.relu(cam)

#         # normalize the heatmap
#         cam /= torch.max(cam)

#         # Calculate weighted combination of feature maps
#         weights = F.adaptive_avg_pool2d(gradients, 1)
#         cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
#         cam = F.relu(cam)

        return cam

    def visualize_gradcam(self, input_tensor, target_size=(64, 64)):
        gradcam = self.generate_gradcam(input_tensor)
        print(f'Shape of the raw gradcam map: {gradcam.shape}')

        gradcam = gradcam.unsqueeze(0).unsqueeze(0)

        # Resize Grad-CAM to match the input image size
        gradcam = F.interpolate(gradcam, size=target_size, mode='bilinear', align_corners=False)

        print(f'Shape of the interpolated gradcam map: {gradcam.shape}')

        # Convert to numpy array for visualization
        gradcam = gradcam.squeeze().cpu().detach().numpy()

        # Normalize for visualization
        gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam) + 1e-8)

        # Display the original image
        original_image = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image) + 1e-8)
        plt.imshow(original_image)

        # Overlay Grad-CAM on the original image
        plt.imshow(gradcam, cmap='jet', alpha=0.3, interpolation='bilinear')
        plt.show()
