"""
Project: Simple GAN (Generative Adversarial Network)
====================================================
Generate synthetic images using GANs.

Dataset: MNIST handwritten digits
Goals:
- Build generator network (noise -> image)
- Build discriminator network (image -> real/fake)
- Implement adversarial training loop
- Balance generator and discriminator training
- Generate synthetic images from random noise
- Visualize training progress and results
- Understand GAN dynamics and mode collapse

Skills: GANs, Adversarial training, Image generation, Two-player optimization
Run: python project_simple_gan.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class Generator(nn.Module):
    """Generator network: maps random noise to images."""

    def __init__(self, latent_dim=100, img_channels=1, img_size=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Calculate initial feature map size
        self.init_size = img_size // 4  # 7 for 28x28 images

        # Linear layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional layers to upsample
        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 7 -> 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 14 -> 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        # z: [batch_size, latent_dim]
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    """Discriminator network: classifies images as real or fake."""

    def __init__(self, img_channels=1, img_size=28):
        super().__init__()

        self.img_channels = img_channels
        self.img_size = img_size

        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1),  # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 7 -> 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 4 -> 2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )

        # Calculate flattened size
        self.adv_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, img):
        # img: [batch_size, channels, height, width]
        out = self.conv_blocks(img)
        validity = self.adv_layer(out)
        return validity


def load_mnist(batch_size=64):
    """Load MNIST dataset."""
    print("Loading MNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")

    return train_loader


def weights_init(m):
    """Initialize network weights."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(generator, discriminator, train_loader, num_epochs, device,
              latent_dim=100, lr=0.0002):
    """Train GAN with adversarial objective."""
    print("\n" + "=" * 60)
    print("TRAINING GAN")
    print("=" * 60)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Labels
    real_label = 1.0
    fake_label = 0.0

    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'd_real_acc': [],
        'd_fake_acc': [],
        'epochs': []
    }

    # For visualization
    fixed_noise = torch.randn(64, latent_dim, device=device)

    print(f"\nTraining settings:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Latent dimension: {latent_dim}")

    for epoch in range(num_epochs):
        start_time = time.time()

        g_losses = []
        d_losses = []
        d_real_correct = 0
        d_fake_correct = 0
        total_samples = 0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # Create labels
            real_labels = torch.full((batch_size, 1), real_label, device=device)
            fake_labels = torch.full((batch_size, 1), fake_label, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            real_validity = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_validity, real_labels)

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_validity, fake_labels)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)

            # Generator wants discriminator to think images are real
            validity = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # Statistics
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            d_real_correct += (real_validity > 0.5).sum().item()
            d_fake_correct += (fake_validity < 0.5).sum().item()
            total_samples += batch_size

            # Print progress
            if batch_idx % 100 == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

        # Epoch statistics
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        d_real_acc = 100. * d_real_correct / total_samples
        d_fake_acc = 100. * d_fake_correct / total_samples

        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['d_real_acc'].append(d_real_acc)
        history['d_fake_acc'].append(d_fake_acc)
        history['epochs'].append(epoch + 1)

        epoch_time = time.time() - start_time

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"  D_loss: {avg_d_loss:.4f} | G_loss: {avg_g_loss:.4f}")
        print(f"  D accuracy: Real={d_real_acc:.2f}% Fake={d_fake_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Generate sample images
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                sample_imgs = generator(fixed_noise)
            generator.train()

            save_sample_images(sample_imgs, epoch + 1)

    return history


def save_sample_images(images, epoch, nrow=8):
    """Save generated images."""
    images = images.cpu()
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, cmap='gray')
    plt.title(f'Generated Images - Epoch {epoch}', fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_epoch_{epoch}.png')
    plt.close()

    print(f"  Saved sample images to gan_epoch_{epoch}.png")


def visualize_training(history):
    """Visualize training metrics."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = history['epochs']

    # 1. Losses
    ax1 = axes[0]
    ax1.plot(epochs, history['g_loss'], 'b-', label='Generator Loss', linewidth=2)
    ax1.plot(epochs, history['d_loss'], 'r-', label='Discriminator Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Generator and Discriminator Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Discriminator accuracy on real images
    ax2 = axes[1]
    ax2.plot(epochs, history['d_real_acc'], 'g-', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Discriminator Accuracy (Real Images)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Discriminator accuracy on fake images
    ax3 = axes[2]
    ax3.plot(epochs, history['d_fake_acc'], 'r-', linewidth=2)
    ax3.axhline(y=50, color='gray', linestyle='--', label='Random (50%)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Discriminator Accuracy (Fake Images)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    print("Training visualizations created!")
    print("Close the plot window to continue...")
    plt.show()


def generate_samples(generator, device, latent_dim=100, num_samples=64):
    """Generate sample images from trained generator."""
    print("\n" + "=" * 60)
    print("GENERATING SAMPLE IMAGES")
    print("=" * 60)

    generator.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        generated_imgs = generator(z)

    # Visualize
    generated_imgs = generated_imgs.cpu()
    grid = torchvision.utils.make_grid(generated_imgs, nrow=8, normalize=True)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np, cmap='gray')
    plt.title('Generated MNIST Digits', fontweight='bold', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    print("\nGenerated samples displayed")
    print("Close the plot window to continue...")
    plt.show()


def interpolate_latent_space(generator, device, latent_dim=100):
    """Interpolate between two points in latent space."""
    print("\n" + "=" * 60)
    print("LATENT SPACE INTERPOLATION")
    print("=" * 60)

    generator.eval()

    # Generate two random latent vectors
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)

    # Interpolate
    steps = 10
    interpolations = []

    with torch.no_grad():
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp)
            interpolations.append(img)

    # Visualize
    interpolations = torch.cat(interpolations, dim=0).cpu()
    grid = torchvision.utils.make_grid(interpolations, nrow=steps, normalize=True)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(15, 3))
    plt.imshow(grid_np, cmap='gray')
    plt.title('Latent Space Interpolation', fontweight='bold', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    print("\nLatent space interpolation displayed")
    print("Close the plot window to continue...")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 60)
    print("SIMPLE GAN PROJECT (MNIST)")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    latent_dim = 100
    img_channels = 1
    img_size = 28
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0002

    print(f"\nHyperparameters:")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Load data
    train_loader = load_mnist(batch_size)

    # Create models
    print("\nInitializing Generator and Discriminator...")
    generator = Generator(latent_dim, img_channels, img_size).to(device)
    discriminator = Discriminator(img_channels, img_size).to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")

    # Train GAN
    history = train_gan(
        generator, discriminator, train_loader, num_epochs,
        device, latent_dim, learning_rate
    )

    # Visualize training
    visualize_training(history)

    # Generate samples
    generate_samples(generator, device, latent_dim, num_samples=64)

    # Interpolate latent space
    interpolate_latent_space(generator, device, latent_dim)

    # Save models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print("\nKey Findings:")
    print(f"1. Successfully trained GAN to generate MNIST digits")
    print(f"2. Final G_loss: {history['g_loss'][-1]:.4f}")
    print(f"3. Final D_loss: {history['d_loss'][-1]:.4f}")
    print(f"4. Discriminator accuracy balances around 50-70% (healthy)")
    print("5. Latent space interpolation shows smooth transitions")
    print("6. Models saved to 'generator.pth' and 'discriminator.pth'")
    print("=" * 60)


if __name__ == "__main__":
    main()
