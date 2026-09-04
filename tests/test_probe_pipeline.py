import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import layerwise_cv_auc, make_probe_pipeline


def test_probe_has_scaler_inside_pipeline_and_fixed_classifier_settings():
    pipeline = make_probe_pipeline(seed=42)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.C == 1.0
    assert classifier.l1_ratio == 0.0
    assert classifier.max_iter == 2000
    assert classifier.random_state == 42


def test_layerwise_cv_selects_known_signal_layer():
    rng = np.random.default_rng(42)
    n_examples, n_layers, hidden_dim = 400, 6, 12
    labels = np.tile([0, 1], n_examples // 2)
    folds = np.tile(np.arange(5), n_examples // 5)
    activations = rng.normal(size=(n_examples, n_layers, hidden_dim))
    signal_layer = 3
    activations[:, signal_layer, 0] += np.where(labels == 1, 5.0, -5.0)

    fold_aucs, mean_aucs, selected_layer = layerwise_cv_auc(
        activations, labels, folds, seed=42
    )

    assert fold_aucs.shape == (n_layers, 5)
    assert mean_aucs.shape == (n_layers,)
    assert selected_layer == signal_layer
    assert mean_aucs[signal_layer] > 0.99
