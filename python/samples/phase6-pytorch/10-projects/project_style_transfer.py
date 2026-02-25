"""
Project: Neural Style Transfer
==============================
Transfer artistic style from one image to another using pretrained VGG.

Dataset: Synthetic content and style images
Goals:
- Generate synthetic content and style images
- Load pretrained VGG19 for feature extraction
- Extract content features from deep layers
- Extract style features via Gram matrices
- Optimize generated image with combined loss
- Create artistic images through iterative optimization
- Visualize transformation process

Skills: CNNs, Feature extraction, Optimization, Transfer learning
Run: python project_style_transfer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def generate_content_image(size=256):
    """Generate a simple content image with geometric shapes."""
    img = Image.new('RGB', (size, size), color=(230, 240, 255))
    draw = ImageDraw.Draw(img)

    # Draw some shapes
    # Circle
    draw.ellipse([60, 60, 140, 140], fill=(100, 150, 200), outline=(50, 100, 150), width=3)

    # Rectangle
    draw.rectangle([140, 100, 220, 180], fill=(200, 100, 100), outline=(150, 50, 50), width=3)

    # Triangle
    draw.polygon([(180, 40), (240, 120), (120, 120)], fill=(100, 200, 100), outline=(50, 150, 50))

    return img


def generate_style_image(size=256, style_type='waves'):
    """Generate a synthetic style image with patterns."""
    img_array = np.zeros((size, size, 3), dtype=np.uint8)

    if style_type == 'waves':
        # Create wave pattern
        for i in range(size):
            for j in range(size):
                r = int(127 + 127 * np.sin(i * 0.1) * np.cos(j * 0.1))
                g = int(127 + 127 * np.sin(i * 0.15 + 1))
                b = int(127 + 127 * np.cos(j * 0.15 + 2))
                img_array[i, j] = [r, g, b]

    elif style_type == 'stripes':
        # Create diagonal stripes
        for i in range(size):
            for j in range(size):
                if (i + j) % 20 < 10:
                    img_array[i, j] = [255, 100, 100]
                else:
                    img_array[i, j] = [100, 100, 255]

    elif style_type == 'dots':
        # Create dot pattern
        img_array.fill(200)
        for i in range(0, size, 20):
            for j in range(0, size, 20):
                y, x = np.ogrid[:size, :size]
                mask = (x - j) ** 2 + (y - i) ** 2 <= 25
                img_array[mask] = [np.random.randint(50, 255) for _ in range(3)]

    img = Image.fromarray(img_array)
    return img


def preprocess_image(image, size=256):
    """Preprocess image for VGG."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def deprocess_image(tensor):
    """Convert tensor back to displayable image."""
    image = tensor.clone().squeeze(0)

    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean

    # Clamp to valid range
    image = torch.clamp(image, 0, 1)

    # Convert to numpy
    image = image.permute(1, 2, 0).cpu().numpy()

    return image


class VGGFeatures(nn.Module):
    """VGG19 feature extractor for style transfer."""

    def __init__(self):
        super().__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Layer indices for content and style
        self.content_layers = ['21']  # conv4_2
        self.style_layers = ['0', '5', '10', '19', '28']  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1

        # Build sequential modules
        self.model = nn.Sequential()
        for i, layer in enumerate(vgg.children()):
            self.model.add_module(str(i), layer)

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, layers):
        """Extract features from specified layers."""
        features = {}
        for name, module in self.model.named_children():
            x = module(x)
            if name in layers:
                features[name] = x
        return features


def gram_matrix(features):
    """Compute Gram matrix for style representation."""
    batch_size, channels, height, width = features.size()

    # Reshape features
    features = features.view(batch_size * channels, height * width)

    # Compute Gram matrix
    gram = torch.mm(features, features.t())

    # Normalize
    gram = gram.div(batch_size * channels * height * width)

    return gram


def compute_content_loss(content_features, generated_features):
    """Compute content loss."""
    return torch.mean((content_features - generated_features) ** 2)


def compute_style_loss(style_features, generated_features):
    """Compute style loss using Gram matrices."""
    style_gram = gram_matrix(style_features)
    generated_gram = gram_matrix(generated_features)

    return torch.mean((style_gram - generated_gram) ** 2)


def style_transfer(content_img, style_img, device, num_steps=300,
                   content_weight=1.0, style_weight=1000000.0):
    """Perform neural style transfer."""
    print("\n" + "=" * 60)
    print("PERFORMING STYLE TRANSFER")
    print("=" * 60)

    # Initialize VGG feature extractor
    vgg = VGGFeatures().to(device).eval()

    # Extract content features
    content_features = vgg(content_img, vgg.content_layers)

    # Extract style features
    style_features = vgg(style_img, vgg.style_layers)

    # Initialize generated image (start with content image)
    generated_img = content_img.clone().requires_grad_(True)

    # Optimizer
    optimizer = optim.LBFGS([generated_img])

    # For visualization
    history = {
        'total_loss': [],
        'content_loss': [],
        'style_loss': [],
        'iterations': []
    }

    print(f"\nOptimization settings:")
    print(f"  Steps: {num_steps}")
    print(f"  Content weight: {content_weight}")
    print(f"  Style weight: {style_weight}")
    print(f"  Optimizer: LBFGS")

    step = [0]

    def closure():
        """Closure for LBFGS optimizer."""
        optimizer.zero_grad()

        # Extract features from generated image
        generated_content_features = vgg(generated_img, vgg.content_layers)
        generated_style_features = vgg(generated_img, vgg.style_layers)

        # Compute content loss
        content_loss = 0
        for layer in vgg.content_layers:
            content_loss += compute_content_loss(
                content_features[layer],
                generated_content_features[layer]
            )
        content_loss *= content_weight

        # Compute style loss
        style_loss = 0
        for layer in vgg.style_layers:
            style_loss += compute_style_loss(
                style_features[layer],
                generated_style_features[layer]
            )
        style_loss *= style_weight

        # Total loss
        total_loss = content_loss + style_loss

        # Backward
        total_loss.backward()

        # Record
        step[0] += 1
        if step[0] % 50 == 0 or step[0] == 1:
            print(f"  Step {step[0]:3d}: Total={total_loss.item():.2e}, "
                  f"Content={content_loss.item():.2e}, "
                  f"Style={style_loss.item():.2e}")

            history['total_loss'].append(total_loss.item())
            history['content_loss'].append(content_loss.item())
            history['style_loss'].append(style_loss.item())
            history['iterations'].append(step[0])

        # Clamp generated image
        with torch.no_grad():
            generated_img.clamp_(-2.5, 2.5)

        return total_loss

    # Optimization loop
    start_time = time.time()

    for i in range(num_steps):
        optimizer.step(closure)

        if step[0] >= num_steps:
            break

    elapsed_time = time.time() - start_time

    print(f"\nStyle transfer completed in {elapsed_time:.1f}s")

    return generated_img, history


def visualize_results(content_img, style_img, generated_img, history):
    """Visualize style transfer results."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig = plt.figure(figsize=(16, 10))

    # 1. Content image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(deprocess_image(content_img))
    ax1.set_title('Content Image', fontweight='bold', fontsize=12)
    ax1.axis('off')

    # 2. Style image
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(deprocess_image(style_img))
    ax2.set_title('Style Image', fontweight='bold', fontsize=12)
    ax2.axis('off')

    # 3. Generated image
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(deprocess_image(generated_img))
    ax3.set_title('Generated Image', fontweight='bold', fontsize=12)
    ax3.axis('off')

    # 4. Total loss
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(history['iterations'], history['total_loss'], 'b-', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('Total Loss Over Time', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # 5. Content loss
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(history['iterations'], history['content_loss'], 'g-', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Content Loss')
    ax5.set_title('Content Loss Over Time', fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. Style loss
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(history['iterations'], history['style_loss'], 'r-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Style Loss')
    ax6.set_title('Style Loss Over Time', fontweight='bold')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Visualizations created!")
    print("Close the plot window to continue...")
    plt.show()


def compare_styles(content_img, device, num_steps=200):
    """Compare different style patterns."""
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT STYLES")
    print("=" * 60)

    style_types = ['waves', 'stripes', 'dots']
    results = []

    for style_type in style_types:
        print(f"\nGenerating {style_type} style...")
        style_img_pil = generate_style_image(size=256, style_type=style_type)
        style_img = preprocess_image(style_img_pil).to(device)

        generated_img, _ = style_transfer(
            content_img, style_img, device,
            num_steps=num_steps,
            content_weight=1.0,
            style_weight=1000000.0
        )

        results.append({
            'style_type': style_type,
            'style_img': style_img,
            'generated_img': generated_img
        })

    # Visualize comparison
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for idx, result in enumerate(results):
        # Content (same for all)
        if idx == 0:
            axes[idx, 0].imshow(deprocess_image(content_img))
        else:
            axes[idx, 0].axis('off')

        # Style
        axes[idx, 1].imshow(deprocess_image(result['style_img']))
        axes[idx, 1].set_title(f'{result["style_type"].capitalize()} Style', fontweight='bold')
        axes[idx, 1].axis('off')

        # Generated
        axes[idx, 2].imshow(deprocess_image(result['generated_img']))
        axes[idx, 2].set_title(f'{result["style_type"].capitalize()} Result', fontweight='bold')
        axes[idx, 2].axis('off')

    axes[0, 0].set_title('Content Image', fontweight='bold')
    axes[0, 0].axis('off')

    plt.tight_layout()
    print("\nStyle comparison displayed")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("NEURAL STYLE TRANSFER PROJECT")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate images
    print("\nGenerating content and style images...")
    content_img_pil = generate_content_image(size=256)
    style_img_pil = generate_style_image(size=256, style_type='waves')

    print("Content image: 256x256 with geometric shapes")
    print("Style image: 256x256 with wave pattern")

    # Preprocess
    content_img = preprocess_image(content_img_pil).to(device)
    style_img = preprocess_image(style_img_pil).to(device)

    # Perform style transfer
    num_steps = 300
    content_weight = 1.0
    style_weight = 1000000.0

    generated_img, history = style_transfer(
        content_img, style_img, device,
        num_steps=num_steps,
        content_weight=content_weight,
        style_weight=style_weight
    )

    # Visualize results
    visualize_results(content_img, style_img, generated_img, history)

    # Compare different styles
    compare_styles(content_img, device, num_steps=200)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\nOptimization steps: {num_steps}")
    print(f"Final total loss: {history['total_loss'][-1]:.2e}")
    print(f"Final content loss: {history['content_loss'][-1]:.2e}")
    print(f"Final style loss: {history['style_loss'][-1]:.2e}")

    # Save generated image
    generated_img_np = deprocess_image(generated_img)
    generated_img_pil = Image.fromarray((generated_img_np * 255).astype(np.uint8))
    generated_img_pil.save('generated_style_transfer.png')
    print("\nGenerated image saved to 'generated_style_transfer.png'")

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Successfully transferred artistic style to content image")
    print("2. Gram matrices effectively capture style representation")
    print("3. LBFGS optimizer works well for image optimization")
    print("4. Balance between content and style is controlled by weights")
    print("5. VGG19 features enable high-quality style transfer")
    print("=" * 60)


if __name__ == "__main__":
    main()
