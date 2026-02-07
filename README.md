AACHAL SAWANT
2025A2PS0447P

The model was trained using a 2D Convolutional Neural Network (CNN) architecture optimized for spectral image data.
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song).
Input Shape: (128, 150, 1) (representing frequency, time, and channel).
Optimizer: Adam with a learning rate of 0.0001.
Epochs: 50
Batch Size: 32

To improve model generalization and prevent overfitting, the following augmentations were applied to the raw audio:
Noise Injection: Adding white noise to simulate real-world environments.
Pitch Shifting: Adjusting the pitch without changing the duration.
