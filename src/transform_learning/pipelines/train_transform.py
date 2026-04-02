import torch

from transform_learning.training import TransformTrainer


def run_default_pipeline(device=None) -> TransformTrainer:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertices = torch.tensor([[1.0], 
                             [-0.25], 
                             [0.25], 
                             [-1.0]], dtype=torch.float32) # assume normalized
    trainer = TransformTrainer(vertices, device=device)
    trainer.generate_dataset(
        num_samples=1000,
        eps=0.5,
        n=1,
        sampling_dist=[0.33, 0.33, 0.34],
        batch_size=64,
    )
    trainer.train(
        num_epochs=1000,
        learning_rate=0.001,
        hidden_dim=32,
        num_hidden_dim_layers=2,
        out_dim=2,
        activation_fn = torch.nn.Tanh,
    )
    trainer.evaluate()
    trainer.visualize(metrics=["loss", "error"], show=True)
    return trainer


if __name__ == "__main__":
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {selected_device}")
    trainer = run_default_pipeline(device=selected_device)
