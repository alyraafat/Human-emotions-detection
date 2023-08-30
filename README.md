# Emotion Detection from Images using CNNs

## Project Overview

This repository contains code for emotion detection from images using Convolutional Neural Networks (CNNs). The primary focus of this project is to explore various CNN models and techniques to accurately detect human emotions in images.

## Dataset

The dataset utilized for this project is obtained from Kaggle. The dataset consists of labeled images that portray different human emotions. The diverse nature of the dataset provides a strong foundation for training our emotion detection models.

## Different Techniques Explored

### 1. Simple CNN Models

For starters, I developed simple Convolutional Neural Network (CNN) architectures from scratch. These initial models serve as baseline comparisons for assessing the performance of more intricate architectures.

### 2. LeNet Model

To assess the performance of a classic architecture, I implemented the LeNet model. This architecture, renowned for its historical significance in computer vision, was evaluated for its capability in emotion recognition.

### 3. Custom ResNet34 Model

To delve into deeper architectures, I constructed a custom ResNet34 model using model subclassing. This allowed me to experiment with residual connections and a more complex architecture.

### 4. Data Augmentation Techniques

Effective data augmentation is vital for improving model generalization. I incorporated various augmentation techniques, such as rotations, flips, CutMix, and MixUp, to enhance our models' capacity to handle diverse facial expressions.

### 5. Transfer Learning with MobileNetV2

I harnessed transfer learning by employing the MobileNetV2 model, pretrained on an extensive dataset, as a feature extractor. Fine-tuning was performed on this base model to adapt it to the emotion detection dataset.

### 6. Model Ensembling

To further amplify model performance, I ventured into model ensembling. By combining predictions from the custom ResNet34, LeNet, and fine-tuned MobileNetV2 models, I aimed to leverage the strengths of each individual model.

### 7. Feature Map Visualization

To comprehend the models' learning processes, I embarked on feature map visualization. This insightful analysis involved visualizing the intermediate convolutional layers to gain a deeper understanding of how the models processed image features.

### 8.Handling Class Imbalance with Class Weighting

To address the class imbalance present in the emotion detection dataset, I employed class weighting during model training. Class imbalance occurs when certain emotion classes have significantly fewer samples compared to others. This can lead to biased training.

To mitigate this issue, I used class weighting techniques to assign higher weights to underrepresented classes and lower weights to overrepresented classes during the computation of the loss function. This helps the model give more attention to the minority classes and prevents the dominant classes from overwhelming the learning process.

### 9. Conclusion

This project chronicles my exploration of diverse deep learning models and techniques for emotion detection from images. I conducted comparisons of simple and complex models, delved into augmentation strategies, tried out transfer learning, and implemented model ensembling. The visualization of feature maps shed light on our models' feature learning patterns.
Feel free to explore the code and documentation in this repository to delve deeper into my implementations, experiments, and results!

# Note:
Few drawbacks are that I could not run the models for more than 5-10 epochs due the lack of uninterrupted runtimes as well as the long durations of each epoch and the lack of available vram due to the unavailability of full-time gpus as well as the complex architecture of some models (like ResNet34) so the main focus of the notebook was to improve my tensorflow skills and try out new data augmentation techniques as well as new visualization techniques and building complex models like ResNet34 but not to produce a very accurate model. Moreover, some of the code is commented so that if you would like to experiment them on your own and see what will happen if you run certain models on datasets which are altered differently

# Improvements:
Definitely, the models in this notebook can be improved by:
1. Hyperparameter Tuning: Experiment with different hyperparameters such as learning rate, batch size, and optimizer settings. Utilize techniques like grid search or random search to find the optimal combination of hyperparameters that improves model performance.

2. Learning Rate Scheduling: Implement learning rate scheduling techniques, such as step decay or learning rate annealing, to dynamically adjust the learning rate during training. This can help the model converge faster and avoid overshooting.

3. Early Stopping: Incorporate early stopping mechanisms to halt training when the model's performance on the validation set plateaus or starts deteriorating. This prevents overfitting and saves computational resources.

4. Different Optimizers: Experiment with different optimizers like RMSProp, or SGD with momentum. Different optimizers might lead to faster convergence or better generalization, depending on the dataset and architecture.

5. More Complex Architectures: Explore more complex architectures, such as deeper CNNs or even advanced architectures like Inception, to capture intricate features within the images.

6. Ensemble Variations: Experiment with different combinations of models for ensemble learning. Adjust the weights assigned to each model's predictions to find the optimal ensemble configuration.

7. Custom Loss Functions: Design custom loss functions that are tailored to the specific emotion detection task. This can help the model prioritize certain emotions or penalize misclassifications more effectively.

8. Transfer Learning Variants: Besides MobileNetV2, experiment with other pre-trained models like VGG16, ResNet50, or EfficientNet to observe how different architectures impact the results.
