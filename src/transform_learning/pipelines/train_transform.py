import torch

from transform_learning.training import TransformTrainer
from transform_learning.losses import CompositeLoss, HullProjectionLoss, VICRegLoss, SIGRegLoss

def run_default_pipeline(device=None) -> TransformTrainer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = CompositeLoss([(1.0, HullProjectionLoss(
                                        margin=0.0, # -0.1
                                        n_inner_steps=100,
                                        inner_lr=0.005,)),
                             #(0.1, VICRegLoss(gamma=1.0, var_weight=1.0, cov_weight=1.0)),
                             #(1.0, SIGRegLoss(n_directions=64,)),
                            ])

    vertices = torch.linspace(-1.0, 1.0, steps=20, dtype=torch.float32).unsqueeze(1)
    trainer = TransformTrainer(device=device)
    trainer.generate_dataset(
        num_samples=1000,
        eps=1.0,
        n=2,
        sampling_dist=[0.5, 0.5, 0.0], # [0.5, 0.5, 0.0]
        batch_size=64,
    )

    trainer.train(
        vertices = vertices,
        loss_fn = loss_fn,
        num_epochs = 1000,
        learning_rate = 5e-5,
        hidden_dim = 256,
        num_hidden_dim_layers = 3,
        out_dim = 8,
        activation_fn = torch.nn.GELU,
        weight_decay = 0.0,
    )
    # train() handles Ctrl-C internally and restores the best model, so these
    # finalization steps always run. Save first so the model is persisted even
    # if a later step is interrupted.
    trainer.save_model("results")
    trainer.evaluate()
    trainer.visualize(metrics=["loss", "success"], save_dir="results")
    return trainer


if __name__ == "__main__":
    selected_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {selected_device}")
    trainer = run_default_pipeline(device=selected_device)
