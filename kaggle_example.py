"""
Kaggle Example Script for DSen2-CR Cloud Removal

This script shows how to run DSen2-CR cloud removal on Kaggle.

Setup Instructions for Kaggle:
1. Create a new Kaggle Notebook
2. Add your data as input (cloudy S2 images, SAR S1 images)
3. Add the model checkpoint as input (model_SARcarl.hdf5)
4. Run this script

Required Kaggle Input Datasets:
- Your image data (S2 cloudy + S1 SAR)
- DSen2-CR model checkpoint (.hdf5 file)
"""

# ============================================================================
# STEP 1: Install dependencies (if needed)
# ============================================================================
# Uncomment if you need to install specific versions
# !pip install rasterio scikit-image keras==2.2.4 tensorflow-gpu==1.15.0

# ============================================================================
# STEP 2: Clone the DSen2-CR repository
# ============================================================================
import os
import sys

# Clone the repo (only needed once)
if not os.path.exists('/kaggle/working/dsen2-cr-predict_test-'):
    !git clone https://github.com/mohit-epic/dsen2-cr-predict_test-.git /kaggle/working/dsen2-cr-predict_test-

# Add to Python path
sys.path.insert(0, '/kaggle/working/dsen2-cr-predict_test-/Code')

# ============================================================================
# STEP 3: Import the prediction script
# ============================================================================
from test_dsen2cr_single_image import test_single_image

# ============================================================================
# STEP 4: Set your file paths
# ============================================================================

# IMPORTANT: Update these paths to match your Kaggle input data!

# Path to your cloudy Sentinel-2 image (13 bands)
CLOUDY_IMAGE = '/kaggle/input/your-dataset/ROIs2017_winter_s2_cloudy_102_p100.tif'

# Path to your SAR Sentinel-1 image (2 bands: VV + VH)
SAR_IMAGE = '/kaggle/input/your-dataset/ROIs2017_winter_s1_102_p100.tif'

# Path to the DSen2-CR model checkpoint
MODEL_CHECKPOINT = '/kaggle/input/dsen2cr-model/model_SARcarl.hdf5'

# (Optional) Path to cloud-free reference image for computing metrics
CLOUDFREE_IMAGE = '/kaggle/input/your-dataset/ROIs2017_winter_s2_102_p100.tif'  # or None

# Output directory (results will be saved here)
OUTPUT_DIR = '/kaggle/working/dsen2cr_results'

# ============================================================================
# STEP 5: Run the prediction
# ============================================================================

print("Starting DSen2-CR cloud removal...")

output = test_single_image(
    image_path=CLOUDY_IMAGE,
    sar_path=SAR_IMAGE,
    model_checkpoint=MODEL_CHECKPOINT,
    output_dir=OUTPUT_DIR,
    cloudfree_path=CLOUDFREE_IMAGE,  # Set to None if you don't have reference
    device='cuda'  # Use 'cpu' if GPU is not available
)

print("\n✅ Done! Check the output directory for results:")
print(f"   {OUTPUT_DIR}")

# ============================================================================
# STEP 6: Display results (optional)
# ============================================================================

from IPython.display import Image, display
import matplotlib.pyplot as plt

# Display the RGB visualization
if os.path.exists(f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'):
    display(Image(filename=f'{OUTPUT_DIR}/output_cloudremoved_rgb.png'))

# Display comparison if available
if os.path.exists(f'{OUTPUT_DIR}/comparison.png'):
    print("\nComparison (Input | Output | Reference):")
    display(Image(filename=f'{OUTPUT_DIR}/comparison.png'))

# Display metrics if available
if os.path.exists(f'{OUTPUT_DIR}/metrics.txt'):
    print("\nMetrics:")
    with open(f'{OUTPUT_DIR}/metrics.txt', 'r') as f:
        print(f.read())

# ============================================================================
# EXAMPLE: Process multiple images in a loop
# ============================================================================

"""
# If you have multiple images to process:

import glob

cloudy_images = glob.glob('/kaggle/input/your-dataset/*cloudy*.tif')

for i, cloudy_img in enumerate(cloudy_images):
    # Extract scene ID to find corresponding SAR image
    scene_id = cloudy_img.split('_')[-2] + '_' + cloudy_img.split('_')[-1].replace('.tif', '')
    sar_img = f'/kaggle/input/your-dataset/ROIs2017_winter_s1_{scene_id}.tif'
    
    if not os.path.exists(sar_img):
        print(f"SAR image not found for {cloudy_img}, skipping...")
        continue
    
    output_dir = f'/kaggle/working/results_{i:03d}'
    
    print(f"\nProcessing image {i+1}/{len(cloudy_images)}: {cloudy_img}")
    
    test_single_image(
        image_path=cloudy_img,
        sar_path=sar_img,
        model_checkpoint=MODEL_CHECKPOINT,
        output_dir=output_dir,
        cloudfree_path=None,
        device='cuda'
    )
"""
