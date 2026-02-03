"""
Test a single image with DSen2-CR model (Cloud Removal using SAR-Optical Fusion)

Usage:
    python test_dsen2cr_single_image.py \
        --image_path /path/to/cloudy_image.tif \
        --sar_path /path/to/sar_image.tif \
        --model_checkpoint /path/to/model.hdf5 \
        --output_dir /kaggle/working/dsen2cr_output \
        --cloudfree_path /path/to/reference.tif (optional, for metrics)

Author: Adapted for DSen2-CR architecture
Compatible with: Keras 2.2.4, TensorFlow 1.15.0
"""

import os
import sys
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from skimage.metrics import structural_similarity as ssim

# Keras/TensorFlow imports
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
import tensorflow as tf
from keras.models import load_model

# Add codes directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import DSen2-CR network and metrics
from dsen2cr_network import DSen2CR_model
import tools.image_metrics as img_met

K.set_image_data_format('channels_first')


def configure_tf_session():
    """Configure TensorFlow session for optimal GPU usage"""
    try:
        # Try TensorFlow 2.x first
        import tensorflow.compat.v1 as tf1
        tf1.disable_v2_behavior()
        config = tf1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf1.Session(config=config)
        tf1.keras.backend.set_session(sess)
        print("✓ TensorFlow session configured (TF 2.x)")
    except:
        try:
            # Fallback to TF 1.x
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            K.tensorflow_backend.set_session(tf.Session(config=config))
            print("✓ TensorFlow session configured (TF 1.x)")
        except:
            # Skip configuration - not critical
            print("ℹ Using default TensorFlow settings (session config skipped)")


def load_tiff_image(image_path):
    """
    Load TIFF image and ensure proper shape (C, H, W)
    
    Args:
        image_path: Path to TIFF file
        
    Returns:
        image: numpy array with shape (channels, height, width)
    """
    image = tifffile.imread(image_path)
    
    # Ensure proper shape: (channels, height, width)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        # Check if it's (H, W, C) and convert to (C, H, W)
        if image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
            image = np.transpose(image, (2, 0, 1))
    
    # Handle NaN values
    image[np.isnan(image)] = np.nanmean(image)
    
    return image.astype('float32')


def normalize_sar_image(image, max_val_sar=2):
    """
    Normalize SAR image according to DSen2-CR preprocessing
    
    Args:
        image: SAR image with 2 channels (VV, VH)
        max_val_sar: Maximum value for normalization (default: 2)
        
    Returns:
        normalized: Normalized SAR image in range [0, max_val_sar]
    """
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
    """
    Normalize optical image according to DSen2-CR preprocessing
    
    Args:
        image: Optical image with 13 Sentinel-2 bands
        scale: Scale factor (default: 2000)
        
    Returns:
        normalized: Normalized optical image
    """
    clip_min = [0] * 13
    clip_max = [10000] * 13
    
    normalized = np.zeros_like(image)
    for channel in range(len(image)):
        data = image[channel]
        data = np.clip(data, clip_min[channel], clip_max[channel])
        normalized[channel] = data / scale
    
    return normalized


def denormalize_optical_image(image, scale=2000):
    """
    Denormalize optical image back to original scale
    
    Args:
        image: Normalized optical image
        scale: Scale factor used during normalization
        
    Returns:
        denormalized: Image in original scale [0, 10000]
    """
    return np.clip(image * scale, 0, 10000).astype('float32')


def calculate_psnr(pred, ref, scale=2000):
    """
    Calculate PSNR between predicted and reference images
    Following DSen2-CR metric computation
    
    Args:
        pred: Predicted image (C, H, W) normalized [0, 1]
        ref: Reference image (C, H, W) normalized [0, 1]
        scale: Scale factor for denormalization
    
    Returns:
        PSNR value in dB
    """
    # Denormalize for PSNR calculation (as done in DSen2-CR)
    pred_scaled = pred * scale
    ref_scaled = ref * scale
    
    mse = np.mean((pred_scaled - ref_scaled) ** 2)
    
    if mse == 0:
        return float('inf')
    
    rmse = np.sqrt(mse)
    psnr = 20.0 * np.log10(10000.0 / rmse)
    
    return psnr


def calculate_ssim(pred, ref, scale=2000):
    """
    Calculate SSIM between predicted and reference images
    
    Args:
        pred: Predicted image (C, H, W) normalized
        ref: Reference image (C, H, W) normalized
        scale: Scale factor for denormalization
    
    Returns:
        SSIM value (mean across channels)
    """
    # Denormalize
    pred_scaled = pred * scale
    ref_scaled = ref * scale
    
    if len(pred.shape) == 3:
        ssim_values = []
        for c in range(pred.shape[0]):
            ssim_val = ssim(ref_scaled[c], pred_scaled[c], data_range=10000.0)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return ssim(ref_scaled, pred_scaled, data_range=10000.0)


def calculate_sam(pred, ref):
    """
    Calculate Spectral Angle Mapper (SAM)
    
    Args:
        pred: Predicted image (C, H, W)
        ref: Reference image (C, H, W)
    
    Returns:
        SAM value in degrees
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    ref_flat = ref.reshape(ref.shape[0], -1)
    
    dots = np.sum(pred_flat * ref_flat, axis=0)
    norms_pred = np.linalg.norm(pred_flat, axis=0)
    norms_ref = np.linalg.norm(ref_flat, axis=0)
    
    valid = (norms_pred > 1e-8) & (norms_ref > 1e-8)
    
    norms_prod = norms_pred[valid] * norms_ref[valid]
    dots_valid = dots[valid]
    
    cos_angles = np.clip(dots_valid / norms_prod, -1, 1)
    angles = np.arccos(cos_angles)
    
    sam_degrees = np.degrees(np.mean(angles))
    return sam_degrees


def calculate_mae(pred, ref):
    """
    Calculate Mean Absolute Error
    
    Args:
        pred: Predicted image (C, H, W)
        ref: Reference image (C, H, W)
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(pred - ref))


def calculate_rmse(pred, ref):
    """
    Calculate Root Mean Square Error
    
    Args:
        pred: Predicted image (C, H, W)
        ref: Reference image (C, H, W)
    
    Returns:
        RMSE value
    """
    mse = np.mean((pred - ref) ** 2)
    return np.sqrt(mse)


def find_reference_image(image_path):
    """
    Find the corresponding cloud-free reference image for a cloudy image
    
    Args:
        image_path: Path to cloudy optical image
    
    Returns:
        Path to reference image if found, None otherwise
    """
    base_path = image_path.replace('.tif', '').replace('.TIF', '')
    filename_base = os.path.basename(base_path)
    
    # Extract scene ID
    scene_id_parts = filename_base.split('_')
    if len(scene_id_parts) >= 2:
        scene_id = '_'.join(scene_id_parts[-2:])
    else:
        scene_id = filename_base
    
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)
    
    ref_candidates = [
        os.path.join(image_dir, f'ROIs2017_winter_s2_{scene_id}.tif'),
        os.path.join(parent_dir, 'ROIs2017_winter_s2_cloudfree', f'*{scene_id}*.tif'),
    ]
    
    for candidate in ref_candidates:
        if '*' in candidate:
            matches = glob.glob(candidate)
            if matches:
                return matches[0]
        elif os.path.exists(candidate):
            return candidate
    
    return None


def create_rgb_visualization(image_data, scale=2000, brighten_limit=2000):
    """
    Create RGB visualization from 13-band Sentinel-2 image
    Uses bands 4, 3, 2 (Red, Green, Blue) for true color
    
    Args:
        image_data: Image array (13, H, W) in original scale [0, 10000]
        scale: Scale factor
        brighten_limit: Brightness limit for visualization
        
    Returns:
        rgb: RGB image (H, W, 3) as uint8
    """
    # Extract RGB bands (4, 3, 2 -> indices 3, 2, 1)
    r = image_data[3]
    g = image_data[2]
    b = image_data[1]
    
    # Clip to brighten limit
    r = np.clip(r, 0, brighten_limit)
    g = np.clip(g, 0, brighten_limit)
    b = np.clip(b, 0, brighten_limit)
    
    # Stack and normalize
    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)
    
    if np.nanmax(rgb) == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / np.nanmax(rgb))
    
    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    
    return rgb.astype(np.uint8)


def test_single_image(image_path, sar_path, model_checkpoint, output_dir, 
                      cloudfree_path=None, device='cuda'):
    """
    Test a single image with the DSen2-CR model
    
    Args:
        image_path: Path to the cloudy optical image (S2, 13 bands)
        sar_path: Path to SAR image (S1, 2 bands VV+VH)
        model_checkpoint: Path to the model checkpoint (.hdf5)
        output_dir: Directory to save output images
        cloudfree_path: Path to cloud-free reference image (optional, for metrics)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        output_image: Predicted cloud-free optical image
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("DSen2-CR: Cloud Removal using Deep Residual Network and SAR-Optical Fusion")
    print("="*80)
    
    # Configure TensorFlow session
    configure_tf_session()
    
    # Validate input files
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Optical image not found: {image_path}")
    if not os.path.exists(sar_path):
        raise FileNotFoundError(f"SAR image not found: {sar_path}")
    if not os.path.exists(model_checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
    
    print(f"\n📂 Input Files:")
    print(f"  Optical (S2): {image_path}")
    print(f"  SAR (S1):     {sar_path}")
    print(f"  Checkpoint:   {model_checkpoint}")
    
    # Load images
    print(f"\n📥 Loading images...")
    sar_data = load_tiff_image(sar_path)
    optical_data = load_tiff_image(image_path)
    
    print(f"  SAR shape:     {sar_data.shape}")
    print(f"  Optical shape: {optical_data.shape}")
    
    # Validate shapes
    if sar_data.shape[0] != 2:
        raise ValueError(f"SAR image must have 2 channels (VV, VH), got {sar_data.shape[0]}")
    if optical_data.shape[0] != 13:
        raise ValueError(f"Optical image must have 13 channels, got {optical_data.shape[0]}")
    
    # Get spatial dimensions
    _, H, W = optical_data.shape
    
    # Normalize images
    print(f"\n🔧 Preprocessing...")
    sar_normalized = normalize_sar_image(sar_data, max_val_sar=2)
    optical_normalized = normalize_optical_image(optical_data, scale=2000)
    
    print(f"  SAR range:     [{sar_normalized.min():.4f}, {sar_normalized.max():.4f}]")
    print(f"  Optical range: [{optical_normalized.min():.4f}, {optical_normalized.max():.4f}]")
    
    # Prepare inputs (add batch dimension)
    input_opt = np.expand_dims(optical_normalized, axis=0)  # (1, 13, H, W)
    input_sar = np.expand_dims(sar_normalized, axis=0)      # (1, 2, H, W)
    
    # Load model
    print(f"\n🧠 Loading DSen2-CR model...")
    
    # Define custom objects for metrics
    custom_objects = {
        'carl_error': img_met.carl_error,
        'cloud_mean_absolute_error': img_met.cloud_mean_absolute_error,
        'cloud_mean_squared_error': img_met.cloud_mean_squared_error,
        'cloud_mean_sam': img_met.cloud_mean_sam,
        'cloud_mean_absolute_error_clear': img_met.cloud_mean_absolute_error_clear,
        'cloud_psnr': img_met.cloud_psnr,
        'cloud_root_mean_squared_error': img_met.cloud_root_mean_squared_error,
        'cloud_bandwise_root_mean_squared_error': img_met.cloud_bandwise_root_mean_squared_error,
        'cloud_mean_absolute_error_covered': img_met.cloud_mean_absolute_error_covered,
        'cloud_ssim': img_met.cloud_ssim,
        'cloud_mean_sam_covered': img_met.cloud_mean_sam_covered,
        'cloud_mean_sam_clear': img_met.cloud_mean_sam_clear,
    }
    
    try:
        model = load_model(model_checkpoint, custom_objects=custom_objects)
        print(f"  ✓ Model loaded successfully from checkpoint")
    except Exception as e:
        print(f"  ⚠ Error loading model: {e}")
        print(f"  Attempting to build model architecture and load weights...")
        
        # Build model architecture
        input_shape = ((13, H, W), (2, H, W))
        model, _ = DSen2CR_model(
            input_shape,
            batch_per_gpu=1,
            num_layers=16,
            feature_size=256,
            use_cloud_mask=True,
            include_sar_input=True
        )
        
        # Load weights
        model.load_weights(model_checkpoint)
        print(f"  ✓ Model weights loaded successfully")
    
    # Run inference
    print(f"\n🚀 Running inference...")
    prediction = model.predict([input_opt, input_sar], batch_size=1)
    
    # Extract predicted image (remove batch dimension and extra channel if present)
    # DSen2-CR outputs (1, 14, H, W) where last channel is cloud mask
    # We only need the first 13 channels
    output_normalized = prediction[0, 0:13, :, :]  # (13, H, W)
    
    print(f"  ✓ Prediction complete")
    print(f"  Output shape: {output_normalized.shape}")
    print(f"  Output range: [{output_normalized.min():.4f}, {output_normalized.max():.4f}]")
    
    # Denormalize output
    output_denormalized = denormalize_optical_image(output_normalized, scale=2000)
    
    print(f"  Denormalized range: [{output_denormalized.min():.1f}, {output_denormalized.max():.1f}]")
    
    # Compute metrics if reference is available
    print("\n" + "="*80)
    print("📊 Quality Metrics")
    print("="*80)
    
    ref_path = cloudfree_path
    if ref_path is None:
        ref_path = find_reference_image(image_path)
    
    if ref_path and os.path.exists(ref_path):
        try:
            print(f"Reference image: {ref_path}")
            ref_image = load_tiff_image(ref_path)
            ref_normalized = normalize_optical_image(ref_image, scale=2000)
            
            # Calculate metrics
            psnr = calculate_psnr(output_normalized, ref_normalized, scale=2000)
            ssim_val = calculate_ssim(output_normalized, ref_normalized, scale=2000)
            sam = calculate_sam(output_normalized, ref_normalized)
            mae = calculate_mae(output_normalized, ref_normalized)
            rmse = calculate_rmse(output_normalized, ref_normalized)
            
            print(f"\n  PSNR:  {psnr:.4f} dB")
            print(f"  SSIM:  {ssim_val:.4f}")
            print(f"  SAM:   {sam:.4f}°")
            print(f"  MAE:   {mae:.6f}")
            print(f"  RMSE:  {rmse:.6f}")
            
            # Save metrics to file
            metrics_txt = os.path.join(output_dir, 'metrics.txt')
            with open(metrics_txt, 'w') as f:
                f.write(f"DSen2-CR Cloud Removal Metrics\n")
                f.write(f"Image: {os.path.basename(image_path)}\n")
                f.write("="*50 + "\n")
                f.write(f"PSNR (dB):         {psnr:.4f}\n")
                f.write(f"SSIM:              {ssim_val:.4f}\n")
                f.write(f"SAM (degrees):     {sam:.4f}\n")
                f.write(f"MAE:               {mae:.6f}\n")
                f.write(f"RMSE:              {rmse:.6f}\n")
            print(f"\n  ✓ Metrics saved to: {metrics_txt}")
            
        except Exception as e:
            print(f"  ⚠ Could not calculate metrics: {e}")
    else:
        print("  ℹ Reference image not found. Skipping metric calculation.")
        print("    (Provide --cloudfree_path to specify reference image)")
    
    print("="*80)
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    
    # 1. Save full 13-band TIFF (using tifffile)
    output_tiff_path = os.path.join(output_dir, 'output_cloudremoved_13bands.tif')
    tifffile.imwrite(output_tiff_path, output_denormalized.astype('float32'))
    print(f"  ✓ Saved 13-band TIFF: {output_tiff_path}")
    
    # 2. Save RGB composite TIFF (using tifffile)
    rgb_bands = np.stack([output_denormalized[3], output_denormalized[2], output_denormalized[1]], axis=0)
    rgb_tiff_path = os.path.join(output_dir, 'output_cloudremoved_rgb.tif')
    tifffile.imwrite(rgb_tiff_path, rgb_bands.astype('float32'))
    print(f"  ✓ Saved RGB composite TIFF: {rgb_tiff_path}")
    
    # 3. Create and save RGB visualization
    rgb_viz = create_rgb_visualization(output_denormalized, scale=2000, brighten_limit=2000)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_viz)
    plt.title('DSen2-CR Cloud-Removed Image (RGB True Color)', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    output_png_path = os.path.join(output_dir, 'output_cloudremoved_rgb.png')
    plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"  ✓ Saved PNG visualization: {output_png_path}")
    
    # 4. Save comparison visualization if reference exists
    if ref_path and os.path.exists(ref_path):
        try:
            ref_rgb = create_rgb_visualization(ref_image, scale=2000, brighten_limit=2000)
            cloudy_rgb = create_rgb_visualization(optical_data, scale=2000, brighten_limit=2000)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(cloudy_rgb)
            axes[0].set_title('Input (Cloudy)', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(rgb_viz)
            axes[1].set_title('DSen2-CR Output', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(ref_rgb)
            axes[2].set_title('Reference (Cloud-Free)', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'comparison.png')
            plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"  ✓ Saved comparison: {comparison_path}")
        except Exception as e:
            print(f"  ⚠ Could not create comparison: {e}")
    
    # 5. Save individual band visualizations
    try:
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.flatten()
        
        for i in range(13):
            ax = axes[i]
            band_data = output_denormalized[i]
            im = ax.imshow(band_data, cmap='viridis', vmin=0, vmax=5000)
            ax.set_title(f'Band {i+1}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(13, 15):
            axes[i].axis('off')
        
        plt.suptitle('DSen2-CR Output - All 13 Sentinel-2 Bands', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        bands_viz_path = os.path.join(output_dir, 'output_all_bands.png')
        plt.savefig(bands_viz_path, dpi=150)
        plt.close()
        
        print(f"  ✓ Saved all bands visualization: {bands_viz_path}")
    except Exception as e:
        print(f"  ⚠ Could not create band visualization: {e}")
    
    print("\n" + "="*80)
    print("✅ Processing Complete!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\n📋 Output Files:")
    print("  • output_cloudremoved_13bands.tif - Full 13-band output")
    print("  • output_cloudremoved_rgb.tif - RGB composite (bands 4,3,2)")
    print("  • output_cloudremoved_rgb.png - RGB visualization")
    print("  • output_all_bands.png - All 13 bands visualization")
    if ref_path and os.path.exists(ref_path):
        print("  • comparison.png - Side-by-side comparison")
        print("  • metrics.txt - Quantitative evaluation metrics")
    print("="*80)
    
    return output_denormalized


def main():
    parser = argparse.ArgumentParser(
        description='Test DSen2-CR model on a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_dsen2cr_single_image.py \\
      --image_path /data/cloudy_s2.tif \\
      --sar_path /data/sar_s1.tif \\
      --model_checkpoint /models/model_SARcarl.hdf5 \\
      --output_dir /output/results

For Kaggle:
  python test_dsen2cr_single_image.py \\
      --image_path /kaggle/input/data/cloudy.tif \\
      --sar_path /kaggle/input/data/sar.tif \\
      --model_checkpoint /kaggle/input/model/model_SARcarl.hdf5 \\
      --output_dir /kaggle/working/output
        """
    )
    
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the cloudy optical image (S2, 13 bands, TIFF)')
    parser.add_argument('--sar_path', type=str, required=True,
                        help='Path to the SAR image (S1, 2 bands VV+VH, TIFF)')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to the DSen2-CR model checkpoint (.hdf5 file)')
    parser.add_argument('--cloudfree_path', type=str, default=None,
                        help='Path to cloud-free reference image for metrics (optional)')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/dsen2cr_output',
                        help='Directory to save output images')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on (default: cuda)')
    
    args = parser.parse_args()
    
    # Run test
    test_single_image(
        image_path=args.image_path,
        sar_path=args.sar_path,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        cloudfree_path=args.cloudfree_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
