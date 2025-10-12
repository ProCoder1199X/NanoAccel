# NanoAccel - High-Performance Neural Network Accelerator

A lightweight, GPU-accelerated neural network library built with NumPy and CuPy for maximum performance.

## 🚀 Features

- **GPU Acceleration**: Seamless CPU/GPU operations with automatic fallback
- **Multiple Optimizers**: SGD, Adam, RMSprop, AdaGrad with configurable parameters
- **Advanced Layers**: Dense, Conv2D, MaxPooling2D, Dropout, BatchNorm, Flatten
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Swish
- **Loss Functions**: MSE, CrossEntropy, Binary CrossEntropy, Huber
- **Regularization**: L1, L2, Dropout, Batch Normalization
- **Learning Rate Scheduling**: Step decay, exponential decay, cosine annealing
- **Model Serialization**: Save/load models with full state preservation
- **Mixed Precision Training**: FP16/FP32 support for faster training
- **Data Augmentation**: Built-in image augmentation pipeline
- **Progress Tracking**: Real-time training metrics with progress bars

## 📦 Installation

```bash
pip install numpy
# For GPU support (optional but recommended)
pip install cupy-cuda11x  # Replace with your CUDA version
```

## 🔧 Quick Start

### Basic Neural Network

```python
from nanoaccel import NeuralNetwork, Dense, Dropout
from nanoaccel.activations import ReLU, Softmax
from nanoaccel.optimizers import Adam
from nanoaccel.losses import CrossEntropy

# Create model
model = NeuralNetwork()
model.add(Dense(784, 128))
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))
model.add(Softmax())

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
    loss=CrossEntropy()
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_new)
```

### Convolutional Neural Network

```python
from nanoaccel import CNN
from nanoaccel.layers import Conv2D, MaxPooling2D, Flatten, Dense
from nanoaccel.activations import ReLU, Softmax

# Create CNN
model = CNN(input_shape=(28, 28, 1))
model.add(Conv2D(filters=32, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=2, stride=2))
model.add(Conv2D(filters=64, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(MaxPooling2D(pool_size=2, stride=2))
model.add(Flatten())
model.add(Dense(128))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Softmax())

model.compile(optimizer=Adam(0.001), loss=CrossEntropy())
model.fit(X_train, y_train, epochs=20, batch_size=64)
```

## 📊 Advanced Features

### Learning Rate Scheduling

```python
from nanoaccel.schedulers import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    initial_lr=0.001,
    min_lr=0.00001,
    T_max=50
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CrossEntropy(),
    lr_scheduler=scheduler
)
```

### Mixed Precision Training

```python
model.compile(
    optimizer=Adam(0.001),
    loss=CrossEntropy(),
    mixed_precision=True  # Enable FP16 training
)
```

### Data Augmentation

```python
from nanoaccel.augmentation import ImageAugmentation

augmenter = ImageAugmentation(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

# Apply during training
model.fit(
    X_train, y_train,
    epochs=50,
    augmentation=augmenter
)
```

### Model Serialization

```python
# Save model
model.save('my_model.npz')

# Load model
from nanoaccel import load_model
model = load_model('my_model.npz')
```

## 🎯 Complete Example: MNIST Classification

```python
import numpy as np
from nanoaccel import NeuralNetwork
from nanoaccel.layers import Dense, Dropout, BatchNorm
from nanoaccel.activations import ReLU, Softmax
from nanoaccel.optimizers import Adam
from nanoaccel.losses import CrossEntropy
from nanoaccel.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess data
X_train, y_train = load_mnist_data()
X_train = X_train.reshape(-1, 784) / 255.0
y_train = one_hot_encode(y_train, 10)

# Create model
model = NeuralNetwork()
model.add(Dense(784, 256))
model.add(BatchNorm())
model.add(ReLU())
model.add(Dropout(0.3))
model.add(Dense(256, 128))
model.add(BatchNorm())
model.add(ReLU())
model.add(Dropout(0.3))
model.add(Dense(128, 10))
model.add(Softmax())

# Callbacks
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.npz', save_best_only=True)

# Compile and train
model.compile(
    optimizer=Adam(learning_rate=0.001, weight_decay=0.0001),
    loss=CrossEntropy(),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

## 🏗️ Architecture

### Layer Types

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Dense` | Fully connected layer | `input_dim, output_dim, use_bias` |
| `Conv2D` | 2D convolution | `filters, kernel_size, stride, padding` |
| `MaxPooling2D` | Max pooling | `pool_size, stride` |
| `Dropout` | Dropout regularization | `rate` |
| `BatchNorm` | Batch normalization | `momentum, epsilon` |
| `Flatten` | Flatten multi-dimensional input | None |

### Optimizers

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| `SGD` | Stochastic Gradient Descent | `learning_rate, momentum, nesterov` |
| `Adam` | Adaptive Moment Estimation | `learning_rate, beta1, beta2` |
| `RMSprop` | Root Mean Square Propagation | `learning_rate, decay, epsilon` |
| `AdaGrad` | Adaptive Gradient | `learning_rate, epsilon` |

### Loss Functions

- **MSE**: Mean Squared Error (regression)
- **CrossEntropy**: Categorical Cross-Entropy (multi-class)
- **BinaryCrossEntropy**: Binary Cross-Entropy (binary classification)
- **Huber**: Huber Loss (robust regression)

### Activation Functions

- `ReLU`, `LeakyReLU`, `ELU`, `Swish`
- `Sigmoid`, `Tanh`
- `Softmax`, `Softplus`

## ⚡ Performance Tips

1. **Use GPU**: Ensure CuPy is installed for 10-100x speedup
2. **Batch Size**: Larger batches = faster training (memory permitting)
3. **Mixed Precision**: Enable for 2x faster training on modern GPUs
4. **Learning Rate**: Start with 0.001 and use schedulers
5. **Regularization**: Use dropout (0.2-0.5) and weight decay (0.0001)
6. **Batch Normalization**: Add after Dense/Conv layers for faster convergence

## 🧪 Testing

```bash
python -m pytest tests/
python -m pytest tests/ --benchmark  # Run benchmarks
```

## 📈 Benchmarks

On MNIST (60k samples, 10 epochs):
- **CPU (NumPy)**: ~45 seconds
- **GPU (CuPy)**: ~3 seconds
- **Speedup**: ~15x

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 API Reference

### NeuralNetwork Class

```python
model = NeuralNetwork()
model.add(layer)                    # Add layer
model.compile(optimizer, loss)      # Compile model
model.fit(X, y, epochs, batch_size) # Train model
model.predict(X)                    # Make predictions
model.evaluate(X, y)                # Evaluate model
model.save(path)                    # Save model
model.summary()                     # Print model architecture
```

### Layer Configuration

```python
Dense(input_dim, output_dim, use_bias=True, 
      kernel_initializer='glorot_uniform',
      kernel_regularizer=L2(0.01))

Conv2D(filters=32, kernel_size=3, stride=1, 
       padding='same', activation='relu')

Dropout(rate=0.5, training=True)

BatchNorm(momentum=0.99, epsilon=1e-3)
```

## 📄 License

MIT License - see LICENSE file for details

