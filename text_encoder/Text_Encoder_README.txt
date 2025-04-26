
README for Text Encoder Model

1. Required Datasets

To use the Text Encoder, you will need two types of datasets:

1.1 Image-Text Paired Dataset (img_text):
- This dataset should contain images of skin diseases paired with text descriptions of the disease.
- The images should be stored in the `images/` folder.
- The text descriptions should be stored in an Excel file (`img_text_<split>.xlsx`), where each row has:
  - `image`: The name of the image file (without extension).
  - `text`: A textual description of the skin disease.
  - `label`: The corresponding class label (e.g., "Fungal Infection").

Example structure:
img_text/
    train/
        img_text_train.xlsx
        images/
            1.png
            2.png
            ...
    val/
        img_text_val.xlsx
        images/
            1.png
            2.png
            ...

1.2 Text-Only Dataset (text_only):
- This dataset contains only text descriptions without the paired images.
- It is stored in Excel files (`text_only_<split>_<label>.xlsx`), where each row has:
  - `text`: A textual description of the skin disease.
  - `label`: The corresponding class label.

Example structure:
text_only/
    train/
        text_only_train_Fungal_Infection.xlsx
        text_only_train_Vitiligo.xlsx
        ...
    val/
        text_only_val_Fungal_Infection.xlsx
        text_only_val_Vitiligo.xlsx
        ...

1.3 Synthetic Text Dataset (synthetic):
- This dataset contains synthetic textual descriptions of diseases.
- Similar to the Text-Only dataset, it has an Excel file for each class label.
- The rows contain `text` and `label`.

Example structure:
synthetic/
    train/
        synthetic_Fungal_Infection.xlsx
        synthetic_Vitiligo.xlsx
        ...

1.4 Important Notes:
- The class labels in both datasets should be consistent.
- The dataset should be split into train, validation, and test sets.
- Ensure that the text descriptions are properly cleaned (e.g., no unnecessary whitespace or special characters).

2. Model Description

The Text Encoder uses BERT (specifically, Bio_ClinicalBERT) as the backbone for text feature extraction. The final classification layer is a linear layer that maps the output of BERT’s [CLS] token to the target classes (e.g., skin disease types).

- Input: Tokenized text descriptions (with padding and truncation).
- Output: Class probabilities (logits), which are used for disease classification.

3. Training

To train the model, you need the following steps:

3.1 Prepare the dataset:
- Ensure your dataset is in the correct folder structure as described above.
- Use the TextDS class to load and tokenize the datasets for training, validation, and testing.

3.2 Train the model:
- The TextEncoder class is responsible for defining the architecture (BERT + Linear Layer).
- The model is trained using cross-entropy loss, and the optimizer used is AdamW.

The code provided trains the model using 20 epochs with a learning rate of `1e-4`.

Example training code:
model = TextEncoder(len(classes)).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

3.3 Data augmentation (for text):
- Make sure to include random text augmentation techniques for synthetic data generation and generalization.

3.4 Hyperparameters:
- Batch size: 16 for training, 32 for validation.
- Learning rate: 1e-4.
- Epochs: 20.
- Max sequence length: 256.

4. Testing

After training, you can test the model to evaluate its performance on a separate test dataset:

4.1 Load the test dataset:
- Similar to the training and validation datasets, you’ll load the test dataset using the TextDS class.

4.2 Evaluate the model:
- Run the model in evaluation mode to make predictions on the test data.
- Calculate accuracy by comparing predicted labels against the ground truth labels.

Example testing code:
model.eval()  # Set model to evaluation mode
correct = total = 0
with torch.no_grad():
    for batch in test_dl:
        logits = model(**batch)[0]
        preds = logits.argmax(1)
        correct += (preds == batch['labels']).sum().item()
        total += preds.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

5. Dependencies

To run the Text Encoder model, you need the following libraries:
- PyTorch (torch)
- Transformers (transformers)
- Pandas (pandas)
- NumPy (numpy)
- TQDM (tqdm)
- Openpyxl (openpyxl for reading Excel files)

You can install the required dependencies by running:
pip install torch transformers pandas numpy tqdm openpyxl

6. File Structure Summary

Your project should have the following directory structure:
    /Diagnostic-Model
        /text_encoder  # Folder containing the model code (text_encoder.py)
        /train.py      # Training script
        /test.py       # Testing script
        /final_divided # Dataset folder
            /img_text
                /train
                    img_text_train.xlsx
                    images/
                        1.png
                        2.png
                        ...
                /val
            /text_only
                /train
                    text_only_train_Fungal_Infection.xlsx
                    ...
            /synthetic
                /train
                    synthetic_Fungal_Infection.xlsx
                    ...

7. Additional Notes
- Make sure you follow clean data preprocessing (removing unnecessary spaces, special characters, etc.) for better model performance.
- You can adjust the hyperparameters like learning rate, batch size, and epochs based on your available resources.
- For more details chech the colab link of the text encoder in the main READ ME file
