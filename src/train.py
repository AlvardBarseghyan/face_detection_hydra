import hydra
from omegaconf import DictConfig
import tensorflow as tf
from .data import get_datasets
from .models import get_model

@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    train_ds, valid_ds, test_ds = get_datasets(cfg.data)
    model = get_model(cfg.model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.trainer.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=cfg.trainer.epochs,
        callbacks=callbacks
    )

    model.evaluate(test_ds)

if __name__ == "__main__":
    main()


