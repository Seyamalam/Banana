import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard # type: ignore
import matplotlib.pyplot as plt

from data_loader import load_data_tensorflow, get_class_weights
from model_tensorflow import get_model, count_parameters, get_model_size

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Main training function
def main():
    # Set random seed
    set_seed(42)
    
    # Parameters
    num_epochs = 5
    batch_size = 32
    img_size = 128
    num_classes = 7
    
    # Load data
    train_ds, val_ds = load_data_tensorflow(img_size=img_size, batch_size=batch_size)
    
    # Get class weights to handle imbalanced data
    class_weights = get_class_weights()
    
    # Initialize model
    model = get_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    
    # Print model summary
    model.summary()
    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath='models/best_model_tensorflow.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir='logs/fit',
            histogram_freq=1
        )
    ]
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    
    # Get number of steps per epoch
    train_steps = tf.data.experimental.cardinality(train_ds).numpy()
    val_steps = tf.data.experimental.cardinality(val_ds).numpy()
    
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Save final model
    model.save('models/final_model_tensorflow.h5')
    
    # Save model in TensorFlow Lite format for mobile deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('models/model_tensorflow.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model size: {len(tflite_model) / (1024 * 1024):.2f} MB")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history_tensorflow.png')
    plt.show()
    
    # Print best validation accuracy
    best_val_acc = max(history.history['val_accuracy']) * 100
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main() 