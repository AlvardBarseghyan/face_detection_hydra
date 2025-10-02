import tensorflow as tf

def get_datasets(config):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.path.train,
        image_size=tuple(config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        seed=42
    )
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        config.path.valid,
        image_size=tuple(config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        seed=42
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.path.test,
        image_size=tuple(config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
        seed=42
    )

    normalize = lambda x, y: (x / 255.0, y)

    train_ds = train_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize).cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds, test_ds

