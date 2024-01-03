import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        # Replace backward hook registration with full_backward_hook
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, class_idx, input_image_size):
        # Weighted feature map
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(pooled_gradients.size(0)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        # Convert to numpy and resize
        heatmap = heatmap.cpu().detach().numpy()  # Updated line
        heatmap = cv2.resize(heatmap, input_image_size)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    def apply_heatmap(self, original_image, heatmap, alpha=0.5):
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        superimposed_img = heatmap * alpha + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img

    def __call__(self, input_tensor, class_idx, original_image):
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Zero grads
        self.model.zero_grad()

        # Backward pass with the selected class
        class_score = output[0][class_idx]
        class_score.backward()

        # Generate heatmap
        heatmap = self.generate_heatmap(class_idx, (original_image.shape[1], original_image.shape[0]))

        # Apply heatmap
        superimposed_img = self.apply_heatmap(original_image, heatmap)

        return superimposed_img


