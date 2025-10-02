from src.data import get_datasets
from omegaconf import OmegaConf

def test_datasets_shape():
    cfg = OmegaConf.create({
        "path": {"train": "tests/fake_data/train", "valid": "tests/fake_data/valid", "test": "tests/fake_data/test"},
        "image_size": [224, 224],
        "batch_size": 2
    })
    train, valid, test = get_datasets(cfg)
    batch = next(iter(train))
    x, y = batch
    assert x.shape[1:] == (224, 224, 3)

