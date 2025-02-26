import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def build_lightweight_cnn(input_shape=(128, 128, 3), num_classes=7):
    """
    Build a lightweight CNN model using depthwise separable convolutions
    
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
    
    # Initial convolution layer
    x = layers.Conv2D(16, kernel_size=3, strides=2, padding='same', 
                     use_bias=False, kernel_regularizer=l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Depthwise separable convolution blocks
    x = depthwise_separable_block(x, 32, strides=1, weight_decay=weight_decay)
    x = depthwise_separable_block(x, 64, strides=2, weight_decay=weight_decay)
    x = depthwise_separable_block(x, 128, strides=2, weight_decay=weight_decay)
    x = depthwise_separable_block(x, 128, strides=1, weight_decay=weight_decay)
    
    # Global average pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def depthwise_separable_block(x, filters, strides=1, weight_decay=1e-4):
    """
    Create a depthwise separable convolution block with batch normalization and ReLU
    
    Args:
        x: Input tensor
        filters: Number of output filters
        strides: Stride for the depthwise convolution
        weight_decay: Weight decay factor for regularization
        
    Returns:
        Output tensor after applying the depthwise separable convolution block
    """
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                              use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Pointwise convolution
    x = layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
                     use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x

def get_model(input_shape=(128, 128, 3), num_classes=7):
    """
    Get the compiled model
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        A compiled Keras model
    """
    model = build_lightweight_cnn(input_shape=input_shape, num_classes=num_classes)
    
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