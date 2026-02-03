# DSen2-CR Single Image Prediction

Optimized prediction script for DSen2-CR cloud removal model, compatible with Kaggle.

## Quick Start for Kaggle

### 1. Setup in Kaggle Notebook

```python
# Clone the repository
!git clone https://github.com/mohit-epic/dsen2-cr-predict_test-.git
import sys
sys.path.insert(0, '/kaggle/working/dsen2-cr-predict_test-/Code')
```

### 2. Run Prediction

```python
from test_dsen2cr_single_image import test_single_image

# Set your paths
CLOUDY_IMAGE = '/kaggle/input/your-data/cloudy_s2.tif'  # 13 bands
SAR_IMAGE = '/kaggle/input/your-data/sar_s1.tif'        # 2 bands (VV+VH)
MODEL_CHECKPOINT = '/kaggle/input/model/model_SARcarl.hdf5'
OUTPUT_DIR = '/kaggle/working/output'

# Run prediction
output = test_single_image(
    image_path=CLOUDY_IMAGE,
    sar_path=SAR_IMAGE,
    model_checkpoint=MODEL_CHECKPOINT,
    output_dir=OUTPUT_DIR,
    cloudfree_path=None,  # Optional: add reference image path for metrics
    device='cuda'
)
```

### 3. View Results

Results are saved in `OUTPUT_DIR`:
- `output_cloudremoved_13bands.tif` - Full 13-band output
- `output_cloudremoved_rgb.tif` - RGB composite
- `output_cloudremoved_rgb.png` - Visualization
- `comparison.png` - Side-by-side comparison (if reference provided)
- `metrics.txt` - Evaluation metrics (if reference provided)

## Input Requirements

- **Cloudy Optical Image**: Sentinel-2, 13 bands, TIFF format
- **SAR Image**: Sentinel-1, 2 bands (VV + VH), TIFF format
- **Model Checkpoint**: DSen2-CR trained model (.hdf5 file)

## Model Checkpoint

Download the pre-trained model from:
- [DSen2-CR CARL model](https://drive.google.com/file/d/1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg/view?usp=sharing)

Upload to Kaggle as a dataset input.

## Full Example

See `kaggle_example.py` for a complete working example.

## Citation

If you use this code, please cite:

```bibtex
@article{Meraner2020,
  title = "Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion",
  journal = "ISPRS Journal of Photogrammetry and Remote Sensing",
  volume = "166",
  pages = "333 - 346",
  year = "2020",
  author = "Andrea Meraner and Patrick Ebel and Xiao Xiang Zhu and Michael Schmitt",
}
```
