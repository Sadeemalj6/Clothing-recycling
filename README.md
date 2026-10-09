# ♻️ ReVision | Computer Vision for Sustainable Fashion

**ReVision** is a Computer Vision prototype developed during the **Computer Vision Bootcamp by Saudi Digital Academy (SDA)**. It aims to encourage thoughtful clothing donations by helping identify second-hand clothing that is ready for reuse, needs minor repairs, can be transformed into something new, or is better suited for recycling.

The idea is simple: **give clothing a second chance and help people donate their best, usable items.** By supporting clothing assessment, ReVision explores how AI can contribute to more effective donation decisions and reduce textile waste.

## 💡 Problem & Solution

Large quantities of used clothing require sorting before they can be reused or donated. Manually assessing each item can be time-consuming, and not every item is suitable for direct reuse.

ReVision takes a single clothing image and classifies it into one of four categories. It then displays the predicted class, confidence score, and a simplified reuse decision.

| Predicted Class      | Decision |
| -------------------- | -------- |
| `Donation Ready`     | YES ✅    |
| `Minor Modification` | YES ✅    |
| `Transform`          | YES ✅    |
| `Not Suitable`       | NO ❌     |

The YES/NO result is a simplified prototype decision, not a guarantee that an item is suitable for donation. Items classified as `Minor Modification` or `Transform` may require additional work before reuse.

## 📦 Dataset

* **Source:** [Hugging Face: clothingdatasetsecondhand](https://huggingface.co/datasets/wargoninnovation/clothingdatasetsecondhand)
* **Images loaded:** 30,192
* **Invalid labels removed:** 2
* **Images retained:** 30,190
* **Classes:** 4, mapped from the dataset's `usage` field.

| Class              | Images | Original Usage Labels     |
| ------------------ | -----: | ------------------------- |
| Donation Ready     | 26,885 | Reuse + Export            |
| Minor Modification |    181 | Repair                    |
| Transform          |    197 | Remake                    |
| Not Suitable       |  2,927 | Recycle + Energy recovery |

### Dataset Split

A stratified 70/15/15 split was used with random seed 42.

| Split      |     Images |
| ---------- | ---------: |
| Training   |     21,133 |
| Validation |      4,528 |
| Testing    |      4,529 |
| **Total**  | **30,190** |

## 🔧 Computer Vision Pipeline

The image classification workflow:

1. Load and explore the dataset.
2. Resize images to 224 × 224 pixels.
3. Apply random horizontal flipping and rotation (±10°) during training.
4. Convert images to tensors and normalize using ImageNet mean and standard deviation.
5. Run the image through the trained model.
6. Apply Softmax to obtain class probabilities.
7. Display the predicted category, confidence score, and simplified YES/NO decision.

## 🧠 Models & Experiments

Three model approaches were explored and compared.

| Model                    | Configuration                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Baseline CNN             | Three convolutional layers, trained from scratch using Adam (learning rate 0.001) for 10 epochs                            |
| **ResNet50**             | ImageNet-pretrained model with a new four-class classification head, Adam (learning rate 0.0001), 10 epochs, batch size 32 |
| Vision Transformer (ViT) | `google/vit-base-patch16-224`, AdamW (learning rate 3e-5), 5 epochs                                                        |

### Handling Class Imbalance

The dataset is highly imbalanced, with `Donation Ready` representing most of the images. To reduce the impact of this imbalance, class-weighted loss was used in the ResNet50 and ViT experiments.

The class weights were calculated using the inverse square root of each class's training-set count, then normalized by their mean:

`weight = 1 / sqrt(class_count)`

These weights were applied through `nn.CrossEntropyLoss(weight=...)`.

## 📊 Model Evaluation

Models were evaluated on the held-out test set of 4,529 images. Precision, recall, and F1-score are macro-averaged across the four classes.

| Model        |        Accuracy | Macro Precision |    Macro Recall |        Macro F1 |        Mean IoU |
| ------------ | --------------: | --------------: | --------------: | --------------: | --------------: |
| Baseline CNN |          0.8905 |          0.2226 |          0.2500 |          0.2355 |          0.2226 |
| ResNet50     |          0.8159 |          0.3848 |          0.3977 |          0.3884 |          0.2961 |
| ViT          | To be confirmed | To be confirmed | To be confirmed | To be confirmed | To be confirmed |

> **Key finding:** The Baseline CNN achieved high accuracy but predicted `Donation Ready` for every test image. This demonstrates why accuracy alone can be misleading when working with imbalanced datasets.

ResNet50 achieved better macro-averaged metrics and mean IoU than the baseline CNN, making it a more informative model for this multi-class task.

## ⚠️ Limitations

* **Class imbalance:** `Minor Modification` and `Transform` have very few examples compared with `Donation Ready`.
* **Uneven class performance:** The model struggles to distinguish underrepresented clothing categories reliably.
* **Not Suitable detection:** The recorded evaluation showed that 142 of 439 `Not Suitable` test examples were correctly identified.
* **Overfitting risk:** Validation loss increased after epoch 3 in the recorded training run, suggesting possible overfitting. The last-epoch model was saved.
* **Decision limitations:** A single image may not show all damage, stains, or other issues that affect whether clothing is suitable for donation.
* **Prototype status:** Predictions should be treated as guidance for experimentation, not as a definitive donation assessment.

## 🚀 Run the Project

1. Open `clothing_classifier_resnet50_gradio.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set **Runtime → Change runtime type → GPU**.
3. Run the notebook cells in order. The dataset is downloaded from Hugging Face.
4. Run the final cell to launch the Gradio interface.
5. Upload a clothing image or use the webcam option, if supported by the current notebook configuration.

### Dependencies

`torch`, `torchvision`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `seaborn`, `pillow`, `gradio`

## 🔮 Future Improvements

* Expand and balance the dataset, especially for underrepresented classes.
* Explore additional training strategies and early stopping.
* Improve detection of damage, stains, and clothing condition.
* Evaluate the model on more diverse, real-world images.
* Improve the interface and test camera-based image input.
* Deploy the application publicly and explore potential use in donation workflows.
