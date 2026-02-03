"""
DEBUG SCRIPT FOR DSen2-CR
This script visualizes the INPUTS to check for corruption/loading errors.
"""

import os
import sys
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, Input

K.set_image_data_format('channels_first')

# ============================================================================
# CONFIGURATION
# ============================================================================
CLOUDY_IMAGE = '/kaggle/input/your-dataset/cloudy_s2.tif'
SAR_IMAGE = '/kaggle/input/your-dataset/sar_s1.tif'
MODEL_CHECKPOINT = '/kaggle/input/model/model_SARcarl.hdf5'
OUTPUT_DIR = '/kaggle/working/output'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tiff_image(image_path):
    print(f"Reading {image_path}...")
    image = tifffile.imread(image_path)
    print(f"  Raw shape: {image.shape}")
    
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    elif image.ndim == 3:
        # Check if we need to transpose from (H, W, C) to (C, H, W)
        # We assume channels are smaller than spatial dims
        if image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
            print(f"  Transposing from (H, W, C) to (C, H, W)")
            image = np.transpose(image, (2, 0, 1))
        else:
            print(f"  Assuming already (C, H, W)")
            
    image[np.isnan(image)] = np.nanmean(image)
    return image.astype('float32')

def create_rgb_visualization(image_data, brighten_limit=2000):
    # Expects (C, H, W)
    r = np.clip(image_data[3], 0, brighten_limit)
    g = np.clip(image_data[2], 0, brighten_limit)
    b = np.clip(image_data[1], 0, brighten_limit)
    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)
    if np.nanmax(rgb) > 0:
        rgb = 255 * (rgb / np.nanmax(rgb))
    return rgb.astype(np.uint8)

# ============================================================================
# MODEL (Minimal for testing)
# ============================================================================
def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)
    return Add()([input_l, tmp])

def DSen2CR_model(input_shape, batch_per_gpu=1, num_layers=16, feature_size=256):
    class CloudMaskLayer(Layer):
        def __init__(self, **kwargs):
            super(CloudMaskLayer, self).__init__(**kwargs)
        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[1] + 1, input_shape[2], input_shape[3])
        def call(self, inputs):
            shape = tf.shape(inputs)
            zeros = tf.zeros((shape[0], 1, shape[2], shape[3]), dtype=inputs.dtype)
            return tf.concat([inputs, zeros], axis=1)
    
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])
    x = Concatenate(axis=1)([input_opt, input_sar])
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Add()([x, input_opt])
    x = Concatenate(axis=1)([x, input_opt])
    x = CloudMaskLayer()(x)
    return Model(inputs=[input_opt, input_sar], outputs=x)

# ============================================================================
# MAIN
# ============================================================================
def run_debug():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. LOAD DATA
    print("\n1. CHECKING INPUT DATA...")
    try:
        optical = load_tiff_image(CLOUDY_IMAGE)
        sar = load_tiff_image(SAR_IMAGE)
    except Exception as e:
        print(f"Error loading images: {e}")
        return

    # 2. VISUALIZE INPUTS
    print("\n2. VISUALIZING INPUTS...")
    plt.figure(figsize=(15, 5))
    
    # Optical RGB
    plt.subplot(1, 3, 1)
    rgb_opt = create_rgb_visualization(optical)
    plt.imshow(rgb_opt)
    plt.title(f"Input Optical\nShape: {optical.shape}")
    plt.axis('off')
    
    # SAR Visualization (VV band)
    plt.subplot(1, 3, 2)
    plt.imshow(sar[0], cmap='gray')
    plt.title(f"Input SAR (VV)\nShape: {sar.shape}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/debug_inputs.png')
    plt.show()
    print("✓ Inputs visualized")

    # 3. RUN PREDICTION
    print("\n3. RUNNING MODEL...")
    
    # Normalize
    optical_norm = optical.clip(0, 10000) / 2000.0
    sar_norm = sar.copy() # Simplified norm for debug
    
    model = DSen2CR_model(((13, None, None), (2, None, None)))
    model.load_weights(MODEL_CHECKPOINT)
    
    pred = model.predict([optical_norm[None, ...], sar_norm[None, ...]], verbose=1)
    
    # Visualize Output
    output = pred[0, 0:13] # First 13 channels
    output = output * 2000.0
    
    plt.figure(figsize=(10, 10))
    rgb_out = create_rgb_visualization(output)
    plt.imshow(rgb_out)
    plt.title("Model Output")
    plt.axis('off')
    plt.savefig(f'{OUTPUT_DIR}/debug_output.png')
    plt.show()

if __name__ == '__main__':
    run_debug()
