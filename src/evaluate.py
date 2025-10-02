import tensorflow as tf

def evaluate(model_path, test_ds):
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")

