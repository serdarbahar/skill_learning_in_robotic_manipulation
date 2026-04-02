import torch

from transform_learning.training import TransformTrainer
from transform_learning.losses import CompositeLoss, VertexReconstructionLoss, VolumePreservationLoss


def run_default_pipeline(device=None) -> TransformTrainer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = CompositeLoss([(1.0, VertexReconstructionLoss()),
                             (0.0, VolumePreservationLoss())
                            ])

    vertices = torch.tensor([[1.0], 
                            [-0.25],
                            [0.25],
                            [-1.0]], dtype=torch.float32) # assume normalized
    trainer = TransformTrainer(device=device)
    trainer.generate_dataset(
        num_samples=1000,
        eps=0.5,
        n=2,
        sampling_dist=[0.33, 0.33, 0.34],
        batch_size=64,
    )
    trainer.train(
        vertices=vertices,
        loss_fn=loss_fn,
        num_epochs=1000,
        learning_rate=0.0003,
        hidden_dim = 64,
        num_hidden_dim_layers=3,
        out_dim=2,
        activation_fn = torch.nn.SiLU,
        weight_decay=0.0,
    )
    trainer.evaluate()
    trainer.visualize(metrics=["loss", "success"], save_dir="results")
    return trainer


if __name__ == "__main__":
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {selected_device}")
    trainer = run_default_pipeline(device=selected_device)
