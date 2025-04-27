
# Fusion Head for Multimodal Classification

This project implements a multimodal fusion model for disease classification, combining images and text. The model is built using PyTorch and Transformers. It uses a fusion of vision and language models to predict skin disease labels from both images and text descriptions.

## Project Structure

The project is organized into several files, each handling a specific part of the process:

### 1. `model.py`
This file defines the models used for image encoding, text encoding, and the fusion of these modalities. It includes:

- **EffNet**: A custom implementation of EfficientNet used as the image encoder.
- **FusionMLP**: A multi-layer perceptron used to combine the features from both image and text encoders and predict the final disease class.
- **TextEncoder**: A wrapper around the Bio_ClinicalBERT model from the Hugging Face library for text encoding.

### 2. `dataset.py`
This file contains the `MultiModalDS` class, which handles loading and preprocessing of the dataset. It includes logic to handle four different data types:

- **Image + Text (Paired)**: Pairs of images and their corresponding text descriptions.
- **Image-only**: Images without text descriptions.
- **Text-only**: Text descriptions without images.
- **Synthetic**: Synthetic data that contains text descriptions for diseases.

The dataset is split into training, validation, and test sets.

### 3. `train.py`
This script handles the training process. It:

- Initializes the models (`EffNet`, `FusionMLP`, `TextEncoder`).
- Loads the data and sets up DataLoader for training and validation.
- Defines the training loop and validation steps.
- Saves the best model based on validation accuracy.

You can run this script to train the model.

### 4. `test.py`
This file handles model evaluation. It:

- Loads the best model saved from training.
- Evaluates the model on the test dataset and prints the test accuracy.

### 5. `utils.py`
This file contains helper functions used across different scripts:

- **`collate`**: A function for processing a batch of data, including image and text preprocessing.
- **`save_checkpoint`**: Saves the model checkpoint.
- **`test_accuracy`**: A function for evaluating the model's accuracy.

### 6. `main.py`
This is the main entry point of the project. It calls:

- `train_model()` to start the training process.
- `test_accuracy()` to evaluate the model after training.

### 7. `requirements.txt`
This file contains all the dependencies required to run the project. Use this file to install the necessary libraries:

```bash
pip install -r requirements.txt
```

## Installation Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd fusion_head
   ```

2. **Set up the environment**:
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Mount Google Drive**:
   If you're running this on Google Colab, mount your Google Drive to access datasets and save checkpoints.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Dataset Preparation**:
   - Place your dataset in the correct directory structure:
     - `final_divided/img_text/`
     - `final_divided/image_only/`
     - `final_divided/text_only/`
     - `final_divided/synthetic/`
   - Ensure the data follows the expected format for paired images and text, image-only data, and text-only data.

5. **Train the Model**:
   To train the model, simply run the `train.py` script:
   ```bash
   python train.py
   ```

   The model will be trained for the specified number of epochs and the best model will be saved.

6. **Test the Model**:
   After training, evaluate the model's performance on the test dataset:
   ```bash
   python test.py
   ```

7. **Export the Model**:
   After training, the best model will be exported to the `deploy_bundle` directory, ready for deployment.

## Usage

- **Training**: To train the model, execute the training script, which will automatically handle training the image and text encoders, as well as the fusion model. The training script saves the best performing model based on validation accuracy.
  
- **Testing**: The testing script loads the best model and evaluates it on the test dataset, reporting the accuracy.

- **Deployment**: After training, the model is saved in the `deploy_bundle` directory, which can be used for deployment. It contains the necessary components to load and use the model for inference.


