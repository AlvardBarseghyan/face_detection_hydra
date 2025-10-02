from src.models import build_fc, build_cnn, build_vgg
from omegaconf import OmegaConf

def test_fc_model():
    cfg = OmegaConf.create({"hidden_units":[64], "dropouts":[0.5], "num_classes":10})
    model = build_fc(cfg)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

def test_cnn_model():
    cfg = OmegaConf.create({"filters":[16], "kernel_size":3, "num_classes":10})
    model = build_cnn(cfg)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

def test_vgg_model():
    cfg = OmegaConf.create({"trainable":False, "num_classes":10})
    model = build_vgg(cfg)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

