import torch

from transform_learning.training import TransformTrainer
from transform_learning.losses import CompositeLoss, VertexReconstructionLoss, VolumePreservationLoss


def run_default_pipeline(device=None) -> TransformTrainer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = CompositeLoss([(0.01, VertexReconstructionLoss(
                                        temperature=2.0,
                                        outside_margin=30.0,)),
                             (1.0, VolumePreservationLoss())
                            ])

    vertices = torch.linspace(-1.0, 1.0, steps=5, dtype=torch.float32).unsqueeze(1)
    trainer = TransformTrainer(device=device)
    trainer.generate_dataset(
        num_samples=1000,
        eps=1.0,
        n=2,
        sampling_dist=[0.33, 0.33, 0.34],
        batch_size=64,
    )

    trainer.train(
        vertices=vertices,
        loss_fn=loss_fn,
        num_epochs=1000,
        learning_rate=0.001,
        hidden_dim = 64,
        num_hidden_dim_layers = 3,
        out_dim=2,
        activation_fn = torch.nn.GELU,
        weight_decay= 0.0,
    )
    trainer.evaluate()
    trainer.visualize(metrics=["loss", "success"], save_dir="results")
    return trainer


if __name__ == "__main__":
    selected_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {selected_device}")
    trainer = run_default_pipeline(device=selected_device)
