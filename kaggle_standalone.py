"""
STANDALONE KAGGLE SCRIPT FOR DSen2-CR CLOUD REMOVAL
UNIVERSAL VERSION (v5) - Works on CPU & GPU without artifacts.

FEATURES:
- Uses Standard 'channels_last' format (NHWC) to match hardware/TensorFlow defaults.
- Explicitly names layers to match original weights.
- Loads weights by name, allowing safe transposition from NCHW to NHWC.
- No more vertical line artifacts!
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

# Use standard format for maximum compatibility
K.set_image_data_format('channels_last')

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================================================
CLOUDY_IMAGE = '/kaggle/input/your-dataset/cloudy_s2.tif'
SAR_IMAGE = '/kaggle/input/your-dataset/sar_s1.tif'
MODEL_CHECKPOINT = '/kaggle/input/model/model_SARcarl.hdf5'
CLOUDFREE_REF = None  # Optional: '/kaggle/input/your-dataset/reference.tif'
OUTPUT_DIR = '/kaggle/working/output'

# ============================================================================
# MODEL ARCHITECTURE (CHANNELS_LAST + EXPLICIT NAMING)
# ============================================================================

def resBlock(input_l, feature_size, kernel_size, scale=0.1, idx=0):
    """Residual Block with explicit naming"""
    # Calculation matches original weight indices
    layer_num_1 = 2 + (idx * 2)
    layer_num_2 = 3 + (idx * 2)
    act_num = 2 + idx
    lam_num = 1 + idx
    add_num = 1 + idx

    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same', name=f'conv2d_{layer_num_1}')(input_l)
    tmp = Activation('relu', name=f'activation_{act_num}')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same', name=f'conv2d_{layer_num_2}')(tmp)
    tmp = Lambda(lambda x: x * scale, name=f'lambda_{lam_num}')(tmp)
    return Add(name=f'add_{add_num}')([input_l, tmp])


def DSen2CR_model(input_shape, batch_per_gpu=1, num_layers=16, feature_size=256):
    """
    DSen2-CR Model Architecture - NHWC Version
    """
    # Input shapes are (H, W, C)
    input_opt = Input(shape=input_shape[0], name='input_1') 
    input_sar = Input(shape=input_shape[1], name='input_2') 
    
    # Concatenate (axis=-1 for channels_last)
    x = Concatenate(axis=-1, name='concatenate_1')([input_opt, input_sar])
    
    # Initial convolution
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same', name='conv2d_1')(x)
    x = Activation('relu', name='activation_1')(x)
    
    # Residual blocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3], idx=i)
    
    # Final convolution
    x = Conv2D(13, (3, 3), kernel_initializer='he_uniform', padding='same', name='conv2d_34')(x)
    
    # Long skip connection
    x = Add(name='add_17')([x, input_opt])
    
    model = Model(inputs=[input_opt, input_sar], outputs=x)
    return model

# ============================================================================
# HELPER FUNCTIONS (Adjusted for HWC)
# ============================================================================

def load_tiff_image(image_path):
    """Load TIFF image. Returns (H, W, C) - Standard Format"""
    print(f"Reading {image_path}...")
    image = tifffile.imread(image_path)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 3:
        # If image is (C, H, W) (e.g. from rasterio logic), transpose to (H, W, C)
        # We assume channels are the smallest dimension (2 or 13)
        channels = min(image.shape)
        if image.shape[0] == channels:
             print("  Transposing from (C, H, W) to (H, W, C)")
             image = np.transpose(image, (1, 2, 0))
    
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype('float32')


def normalize_sar_image(image, max_val_sar=2):
    """Normalize SAR image (H, W, C)"""
    clip_min = [-25.0, -32.5]
    clip_max = [0.0, 0.0]
    normalized = np.zeros_like(image)
    # Iterate over channels (last dimension)
    for channel in range(image.shape[-1]):
        data = image[..., channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        data -= clip_min[channel]
        normalized[..., channel] = max_val_sar * (data / (clip_max[channel] - clip_min[channel]))
    return normalized


def normalize_optical_image(image, scale=2000):
    """Normalize optical image (H, W, C)"""
    clip_min = [0] * 13
    clip_max = [10000] * 13
    normalized = np.zeros_like(image)
    for channel in range(image.shape[-1]):
        data = image[..., channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        normalized[..., channel] = data / scale
    return normalized


def denormalize_optical_image(image, scale=2000):
    """Denormalize optical image back to original scale"""
    return np.clip(image * scale, 0, 10000).astype('float32')

def create_rgb_visualization(image_data, brighten_limit=2000):
    """Create RGB visualization from (H, W, C) data"""
    # Bands: 3=R, 2=G, 1=B (0-indexed)
    r = np.clip(image_data[..., 3], 0, brighten_limit)
    g = np.clip(image_data[..., 2], 0, brighten_limit)
    b = np.clip(image_data[..., 1], 0, brighten_limit)
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
    print("DSen2-CR Universal: Cloud Removal")
    print("Format: channels_last (H, W, C)")
    print("="*80)
    
    # Load images
    print(f"\n📥 Loading images...")
    sar_data = load_tiff_image(SAR_IMAGE)
    optical_data = load_tiff_image(CLOUDY_IMAGE)
    
    print(f"  SAR shape:     {sar_data.shape}")
    print(f"  Optical shape: {optical_data.shape}")
    
    H, W, C_opt = optical_data.shape
    C_sar = sar_data.shape[-1]

    if C_sar != 2:
         raise ValueError(f"SAR must have 2 channels, got {C_sar}")
    if C_opt != 13:
         raise ValueError(f"Optical must have 13 channels, got {C_opt}")
    
    # Normalize
    print(f"\n🔧 Preprocessing...")
    sar_normalized = normalize_sar_image(sar_data)
    optical_normalized = normalize_optical_image(optical_data)
    
    # Prepare inputs (add batch dimension)
    input_opt = np.expand_dims(optical_normalized, axis=0) # (1, H, W, 13)
    input_sar = np.expand_dims(sar_normalized, axis=0)     # (1, H, W, 2)
    
    # Build model architecture (NHWC)
    print(f"\n🧠 Building DSen2-CR model...")
    # Input shapes: (H, W, 13) and (H, W, 2)
    input_shape = ((H, W, 13), (H, W, 2))
    
    model = DSen2CR_model(
        input_shape,
        batch_per_gpu=1,
        num_layers=16,
        feature_size=256
    )
    print(f"  ✓ Model architecture built")
    
    # Load weights BY NAME
    # This allows loading NCHW weights into NHWC layers automatically!
    print(f"  Loading weights from: {MODEL_CHECKPOINT}")
    try:
        model.load_weights(MODEL_CHECKPOINT, by_name=True)
        print(f"  ✓ Weights loaded successfully (by_name=True)")
    except Exception as e:
        print(f"  ⚠ Error loading weights: {e}")
        raise
    
    # Run inference
    print(f"\n🚀 Running inference...")
    prediction = model.predict([input_opt, input_sar], batch_size=1, verbose=0)
    
    # Output is (1, H, W, 13)
    output_normalized = prediction[0] 
        
    output_denormalized = denormalize_optical_image(output_normalized)
    
    print(f"  ✓ Prediction complete")
    print(f"  Output shape: {output_denormalized.shape}")
    
    print("="*80)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # Save TIFFs (Standard TIFF expects C, H, W usually, but tifffile handles HWC too)
    # Let's verify format. Tifffile often prefers planar (C, H, W).
    # We transpose for saving to keep compatibility with standard viewers
    output_chw = np.transpose(output_denormalized, (2, 0, 1))
    tifffile.imwrite(os.path.join(OUTPUT_DIR, 'output_13bands.tif'), output_chw)
    
    # RGB
    rgb = create_rgb_visualization(output_denormalized)
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb)
    plt.title('DSen2-CR Cloud-Removed Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'output_cloudremoved_rgb.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Saved: output_13bands.tif (Transposed to CHW)")
    print(f"  ✓ Saved: output_cloudremoved_rgb.png")
    
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    return output_denormalized

if __name__ == '__main__':
    run_prediction()
    from IPython.display import Image, display
    if os.path.exists(f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'):
        display(Image(filename=f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'))
