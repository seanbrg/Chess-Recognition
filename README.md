# Chess Piece Classification via Transfer Learning
### *Overcoming the 50% Plateau in Low-Resolution Image Recognition*

##  Project Overview
This project explores the application of **Transfer Learning** to classify chess pieces from a custom, highly imbalanced dataset. Starting with building a custom model to achieve high results on **CIFAR-10** (90% baseline accuracy), the architecture was surgically modified and fine-tuned to recognize six distinct chess piece classes: **Bishop, King, Knight, Pawn, Queen, and Rook.**

The primary challenge involved adapting a high-capacity architecture to a "Small Data" regime (~1,200 images) at a restricted resolution of **32x32 pixels**.

---

## Model Architecture & Evolution

### Phase 1: Creating the Custom Wide & Deep CNN
Initially, a Fully Connected NN was utilized. An iterative process of **ablation** showed that it is fundamentally flawed for image classification. Therefore a custom CNN was utilized and regularization techniques were applied to tune it to the CIFAR-10 dataset. 


### Phase 2: Transfer Learning Into the Chesspiece Dataset
A custom dataset of chesspieces was assembled out of open Kaggle datasets.
The classification head of the Wide & Deep CNN was replaced to accommodate 6 classes instead of 10.
* **Warmup Strategy:** Frozen backbone, training only the fully connected layers.
* **The "Stall":** Accuracy flatlined at 50% due to the **Domain Gap** - CIFAR-10 features (frogs, trucks) did not translate well to the geometric nuances of chess pieces.

### Phase 3: ResNet-50 Integration
To increase feature extraction capacity, we migrated to **ResNet-152**. 
* **Surgical Head Replacement:** The final `fc` layer was replaced with 
  * `nn.Linear(in_features, 6)`
* **Layer Unfreezing:** To break the performance ceiling, we performed selective unfreezing of `layer4`, allowing the deepest residual blocks to adapt to chess-specific silhouettes.



---

##  Ablation Studies
We performed systematic ablation to identify the "Golden Configuration" for this dataset. This involved isolating specific components to measure their impact on the validation loss:

| Component | Change Made                          | Impact on Performance |
| :--- |:-------------------------------------| :--- |
| **Backbone State** | Frozen ➡ Unfrozen `layer4`           | Accuracy jumped from 50% to 68%. |
| **Regularization** | Weight Decay $1e-4 \rightarrow 2e-5$ | Resolved underfitting; allowed weights to optimize. |
| **Input Noise** | Extreme ➡ Gentle Augmentation        | Prevented "feature shattering" at 32x32 resolution. |
| **Class Balancing** | Uniform ➡ WeightedRandomSampler      | Fixed the "King Deficit" where the rarest class was ignored. |



---

##  Data Challenges & Solutions

### 1. The "King" Minority Problem
The dataset was heavily imbalanced, with the **King** class having only 75 samples compared to 280 **Knights**.
* **Observation:** The model initially ignored Kings entirely to minimize global loss.
* **Solution:** Implemented a `WeightedRandomSampler` in the DataLoader to ensure every training batch had equal representation of all 6 classes.



### 2. The 32x32 Resolution Limit
At low resolutions, "Visual Aliasing" occurs - Bishops and Pawns share nearly identical silhouettes.
* **Solution:** Applied **Gentle Data Augmentation** (limited rotation of 10°, slight color jitter) to maintain the integrity of the few available pixels.

---

##  Final Results & Diagnostics
* **Best Validation Accuracy:** 68% (when baseline guessing accuracy is ~16.6)
* **Key Finding:** The model initially "collapsed" on the King class. By using a **Confusion Matrix**, we identified that the model was misidentifying Kings as Queens.
* **Current Bottleneck:** The $32 \times 32$ resolution. At this scale, the "Cross" on a King or the "Slit" in a Bishop is represented by only 1-3 pixels. The size and quality of the database remains a fundamental problem.

## Prerequisites
* Python 3.12+
* See the `requirements.txt` file
