"""
STANDALONE KAGGLE SCRIPT FOR DSen2-CR CLOUD REMOVAL
CORRECT VERSION (v3) - Matches trained model architecture + Keras 3.x Optimized

GPU REQUIRED - Enable GPU in Kaggle settings before running!
"""

import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Keras/TensorFlow imports
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, Input

K.set_image_data_format('channels_first')

# ============================================================================
# DSen2-CR MODEL ARCHITECTURE (EXACT MATCH + KERAS 3.x FIX)
# ============================================================================

def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Residual Block"""
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)
    return Add()([input_l, tmp])


def DSen2CR_model(input_shape, batch_per_gpu=1, num_layers=16, feature_size=256):
    """
    DSen2-CR Model Architecture - Keras 3.x compatible version
    Outputs 27 channels: 13 predicted bands + 13 input bands + 1 cloud mask
    """
    from keras.layers import Layer
    
    class CloudMaskLayer(Layer):
        """Custom layer to add cloud mask channel"""
        def __init__(self, **kwargs):
            super(CloudMaskLayer, self).__init__(**kwargs)
            
        def compute_output_shape(self, input_shape):
            # input_shape is (batch, channels, height, width)
            # return same shape but with channels + 1
            return (input_shape[0], input_shape[1] + 1, input_shape[2], input_shape[3])
        
        def call(self, inputs):
            # inputs shape: (batch, channels, height, width)
            shape = tf.shape(inputs)
            batch_size = shape[0]
            height = shape[2]
            width = shape[3]
            
            # Create zero mask channel using TensorFlow ops
            zeros = tf.zeros((batch_size, 1, height, width), dtype=inputs.dtype)
            return tf.concat([inputs, zeros], axis=1)
    
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])
    
    # Concatenate optical and SAR inputs
    x = Concatenate(axis=1)([input_opt, input_sar])
    
    # Initial convolution
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)
    
    # Residual blocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])
    
    # Final convolution
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    
    # Long skip connection
    x = Add()([x, input_opt])
    
    # Add cloud mask output - concatenate predicted + input + mask
    x = Concatenate(axis=1)([x, input_opt])  # (batch, 26, H, W)
    x = CloudMaskLayer()(x)  # (batch, 27, H, W)
    
    model = Model(inputs=[input_opt, input_sar], outputs=x)
    return model

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================================================
CLOUDY_IMAGE = '/kaggle/input/your-dataset/cloudy_s2.tif'
SAR_IMAGE = '/kaggle/input/your-dataset/sar_s1.tif'
MODEL_CHECKPOINT = '/kaggle/input/model/model_SARcarl.hdf5'
CLOUDFREE_REF = None  # Optional: '/kaggle/input/your-dataset/reference.tif'
OUTPUT_DIR = '/kaggle/working/output'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tiff_image(image_path):
    """Load TIFF image and ensure proper shape (C, H, W)"""
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        if image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
            image = np.transpose(image, (2, 0, 1))
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype('float32')


def normalize_sar_image(image, max_val_sar=2):
    """Normalize SAR image"""
    clip_min = [-25.0, -32.5]
    clip_max = [0.0, 0.0]
    normalized = np.zeros_like(image)
    for channel in range(len(image)):
        data = image[channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        data -= clip_min[channel]
        normalized[channel] = max_val_sar * (data / (clip_max[channel] - clip_min[channel]))
    return normalized


def normalize_optical_image(image, scale=2000):
    """Normalize optical image"""
    clip_min = [0] * 13
    clip_max = [10000] * 13
    normalized = np.zeros_like(image)
    for channel in range(len(image)):
        data = image[channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        normalized[channel] = data / scale
    return normalized


def denormalize_optical_image(image, scale=2000):
    """Denormalize optical image back to original scale"""
    return np.clip(image * scale, 0, 10000).astype('float32')


def calculate_metrics(pred, ref, scale=2000):
    """Calculate all metrics"""
    pred_scaled = pred * scale
    ref_scaled = ref * scale
    
    # PSNR
    mse = np.mean((pred_scaled - ref_scaled) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20.0 * np.log10(10000.0 / rmse) if mse > 0 else float('inf')
    
    # SSIM
    ssim_values = []
    for c in range(pred.shape[0]):
        ssim_val = ssim(ref_scaled[c], pred_scaled[c], data_range=10000.0)
        ssim_values.append(ssim_val)
    ssim_mean = np.mean(ssim_values)
    
    # SAM
    pred_flat = pred.reshape(pred.shape[0], -1)
    ref_flat = ref.reshape(ref.shape[0], -1)
    dots = np.sum(pred_flat * ref_flat, axis=0)
    norms_pred = np.linalg.norm(pred_flat, axis=0)
    norms_ref = np.linalg.norm(ref_flat, axis=0)
    valid = (norms_pred > 1e-8) & (norms_ref > 1e-8)
    cos_angles = np.clip(dots[valid] / (norms_pred[valid] * norms_ref[valid]), -1, 1)
    sam = np.degrees(np.mean(np.arccos(cos_angles)))
    
    # MAE and RMSE (normalized)
    mae = np.mean(np.abs(pred - ref))
    rmse_norm = np.sqrt(np.mean((pred - ref) ** 2))
    
    return psnr, ssim_mean, sam, mae, rmse_norm


def create_rgb_visualization(image_data, brighten_limit=2000):
    """Create RGB visualization"""
    r = np.clip(image_data[3], 0, brighten_limit)
    g = np.clip(image_data[2], 0, brighten_limit)
    b = np.clip(image_data[1], 0, brighten_limit)
    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)
    if np.nanmax(rgb) > 0:
        rgb = 255 * (rgb / np.nanmax(rgb))
    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    return rgb.astype(np.uint8)


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================

def run_prediction():
    """Run DSen2-CR cloud removal prediction"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("DSen2-CR: Cloud Removal using SAR-Optical Fusion")
    print("MATCHES TRAINED MODEL + KERAS 3.x COMPATIBLE")
    print("="*80)
    
    # Load images
    print(f"\n📥 Loading images...")
    sar_data = load_tiff_image(SAR_IMAGE)
    optical_data = load_tiff_image(CLOUDY_IMAGE)
    
    print(f"  SAR shape:     {sar_data.shape}")
    print(f"  Optical shape: {optical_data.shape}")
    
    if sar_data.shape[0] != 2:
        raise ValueError(f"SAR must have 2 channels, got {sar_data.shape[0]}")
    if optical_data.shape[0] != 13:
        raise ValueError(f"Optical must have 13 channels, got {optical_data.shape[0]}")
    
    # Normalize
    print(f"\n🔧 Preprocessing...")
    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)
    
    # Prepare inputs
    input_opt = np.expand_dims(optical_normalized, axis=0)
    input_sar = np.expand_dims(sar_normalized, axis=0)
    
    # Get image dimensions
    _, H, W = optical_data.shape
    
    # Build model architecture (Keras 3.x compatible)
    print(f"\n🧠 Building DSen2-CR model...")
    input_shape = ((13, H, W), (2, H, W))
    model = DSen2CR_model(
        input_shape,
        batch_per_gpu=1,
        num_layers=16,
        feature_size=256
    )
    print(f"  ✓ Model architecture built")
    
    # Load weights
    print(f"  Loading weights from: {MODEL_CHECKPOINT}")
    try:
        model.load_weights(MODEL_CHECKPOINT)
        print(f"  ✓ Weights loaded successfully")
    except Exception as e:
        print(f"  ⚠ Error loading weights: {e}")
        raise
    
    # Run inference
    print(f"\n🚀 Running inference...")
    prediction = model.predict([input_opt, input_sar], batch_size=1, verbose=0)
    
    # Model outputs (1, 27, H, W): [13 predicted bands, 13 input bands, 1 cloud mask]
    # We want the first 13 channels (the predicted cloud-free image)
    if prediction.shape[1] >= 13:
        output_normalized = prediction[0, 0:13, :, :]
    else:
        output_normalized = prediction[0]
        
    output_denormalized = denormalize_optical_image(output_normalized)
    
    print(f"  ✓ Prediction complete")
    print(f"  Output shape: {output_denormalized.shape}")
    
    # Calculate metrics if reference available
    print("\n" + "="*80)
    print("📊 Quality Metrics")
    print("="*80)
    
    if CLOUDFREE_REF and os.path.exists(CLOUDFREE_REF):
        ref_image = load_tiff_image(CLOUDFREE_REF)
        ref_normalized = normalize_optical_image(ref_image)
        psnr, ssim_val, sam, mae, rmse = calculate_metrics(output_normalized, ref_normalized)
        
        print(f"\n  PSNR:  {psnr:.4f} dB")
        print(f"  SSIM:  {ssim_val:.4f}")
        print(f"  SAM:   {sam:.4f}°")
        print(f"  MAE:   {mae:.6f}")
        print(f"  RMSE:  {rmse:.6f}")
        
        # Save metrics
        with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
            f.write(f"DSen2-CR Metrics\n")
            f.write("="*50 + "\n")
            f.write(f"PSNR (dB):     {psnr:.4f}\n")
            f.write(f"SSIM:          {ssim_val:.4f}\n")
            f.write(f"SAM (degrees): {sam:.4f}\n")
            f.write(f"MAE:           {mae:.6f}\n")
            f.write(f"RMSE:          {rmse:.6f}\n")
    else:
        print("  ℹ No reference image provided")
    
    print("="*80)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Save TIFFs
    tifffile.imwrite(os.path.join(OUTPUT_DIR, 'output_13bands.tif'), output_denormalized)
    rgb_bands = np.stack([output_denormalized[3], output_denormalized[2], output_denormalized[1]], axis=0)
    tifffile.imwrite(os.path.join(OUTPUT_DIR, 'output_rgb.tif'), rgb_bands)
    
    # Save PNG visualization
    rgb_viz = create_rgb_visualization(output_denormalized)
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_viz)
    plt.title('DSen2-CR Cloud-Removed Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'output_cloudremoved_rgb.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Saved: output_13bands.tif")
    print(f"  ✓ Saved: output_rgb.tif")
    print(f"  ✓ Saved: output_cloudremoved_rgb.png")
    
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    return output_denormalized


# ============================================================================
# RUN IT!
# ============================================================================

if __name__ == '__main__':
    output = run_prediction()
    
    # Display image
    from IPython.display import Image, display
    if os.path.exists(f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'):
        display(Image(filename=f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'))
