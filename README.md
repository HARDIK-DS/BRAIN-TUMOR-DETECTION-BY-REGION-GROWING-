# Brain Tumor Detection Application

This application uses region growing segmentation to detect and segment brain tumors from MRI images. The interactive UI allows users to upload medical images, fine-tune segmentation parameters, and visualize results.

## Features

- Upload brain MRI images in various formats (PNG, JPG, DICOM, NIfTI)
- Interactive region growing segmentation algorithm
- Visualization of segmentation results
- Quantitative metrics for tumor analysis
- Adjustable algorithm parameters for optimal results
- Export segmentation results

## How to Use

1. Launch the application:
   ```
   streamlit run app.py
   ```

2. Upload a brain MRI image using the file uploader

3. Adjust the algorithm parameters in the sidebar:
   - **Seed Point Selection**: Choose automatic or manual seed point selection
   - **Threshold**: Adjust the intensity threshold for region growing
   - **Connectivity**: Select 4 or 8 neighboring pixels for the algorithm
   - **Max Iterations**: Set the maximum number of iterations for the algorithm
   - **Smoothing Factor**: Control the amount of pre-processing smoothing

4. For manual seed point selection, click on the suspected tumor area in the image

5. Click "Run Segmentation" to perform the segmentation

6. Review the results and metrics

7. Download the segmentation result using the download button

## Algorithm Overview

The region growing segmentation algorithm works as follows:

1. Start from a seed point (either manually selected or automatically detected)
2. Examine neighboring pixels around the region
3. Add neighboring pixels to the region if their intensity is within a threshold
4. Continue until no more pixels can be added or maximum iterations reached
5. Apply post-processing to clean up the segmentation

## Evaluation Metrics

The application calculates several metrics for the segmented region:

- **Area**: Number of pixels in the segmented region
- **Perimeter**: Length of the boundary of the segmented region
- **Circularity**: Measure of how circular the segmented region is (1.0 = perfect circle)

## Requirements

- Python 3.6+
- Streamlit
- OpenCV
- NumPy
- scikit-image
- SimpleITK
- Matplotlib
- pydicom (for DICOM support)
