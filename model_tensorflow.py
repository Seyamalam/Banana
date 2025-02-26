import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_banana_leaf_cnn(input_shape=(128, 128, 3), num_classes=7):
    """
    Build a CNN model for banana leaf classification using the architecture provided
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    # Weight decay for regularization
    weight_decay = 1e-4
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(32, kernel_size=3, padding='same', 
                     kernel_regularizer=l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fifth convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Sixth convolutional block
    x = layers.Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = layers.Dropout(0.5)(x)  # Increased dropout for better regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def get_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Get the compiled model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    model = build_banana_leaf_cnn(input_shape=input_shape, num_classes=num_classes)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to count parameters
def count_parameters(model):
    return model.count_params()

# Function to calculate model size in MB
def get_model_size(model):
    # Get model size in bytes
    model_bytes = 0
    for weight in model.weights:
        model_bytes += weight.numpy().nbytes
    
    # Convert to MB
    model_mb = model_bytes / (1024 * 1024)
    return model_mb

# Test the model
if __name__ == "__main__":
    model = get_model()
    model.summary()
    
    print(f"Number of parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB") 