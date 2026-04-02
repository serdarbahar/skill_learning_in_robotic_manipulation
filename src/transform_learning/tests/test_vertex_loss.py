# tests/test_vertex_reconstruction_loss.py
import torch
import pytest
from transform_learning.losses.vertex_reconstruction import vertex_reconstruction_loss

class TestVertexReconstructionLoss:

    def setup_method(self):
        """Shared fixtures — simple 2D vertices forming a square."""
        self.vertices = torch.tensor([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ])
        self.outside_margin = 0.1

    def test_inside_point_has_low_loss(self):
        """A point inside the hull with label=1 should produce near-zero loss."""
        outputs = torch.tensor([[10.0, 0.0]])  # centroid
        labels = torch.tensor([1])
        loss = vertex_reconstruction_loss(outputs, self.vertices, labels, outside_margin=self.outside_margin)
        assert loss.item() < 0.01

    def test_outside_point_with_label_0_no_penalty_beyond_margin(self):
        """A point far outside with label=0 should have zero loss if error > margin."""
        outputs = torch.tensor([[50.0, 50.0]])
        labels = torch.tensor([0])
        loss = vertex_reconstruction_loss(outputs, self.vertices, labels, outside_margin=self.outside_margin)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_outside_point_with_label_0_penalized_within_margin(self):
        """A point barely outside with label=0 should be penalized (margin violation)."""
        outputs = torch.tensor([[10.01, 5]])  # just outside
        labels = torch.tensor([0])
        loss = vertex_reconstruction_loss(outputs, self.vertices, labels, outside_margin=self.outside_margin)
        assert loss.item() > 0.0

    def test_inside_point_with_wrong_label_penalized(self):
        """A point inside the hull but labeled 0 — margin should push it out."""
        outputs = torch.tensor([[5, 5]])
        labels = torch.tensor([0])
        loss = vertex_reconstruction_loss(outputs, self.vertices, labels, outside_margin=self.outside_margin)
        assert loss.item() > 0.01

    def test_batch_mixed_labels(self):
        """Batch with both labels — loss should be nonzero."""
        outputs = torch.tensor([[5, 5], [50.0, 50.0]])
        labels = torch.tensor([1, 1])  # inside correct, outside wrong
        loss = vertex_reconstruction_loss(outputs, self.vertices, labels, outside_margin=self.outside_margin)
        assert loss.item() > 0.0

    def test_loss_decreases_as_inside_point_moves_from_vertex_to_centroid(self):
        """Moving an inside-labeled point from a vertex to the centroid should reduce loss."""
        labels = torch.tensor([1])
        loss_vertex = vertex_reconstruction_loss(
            torch.tensor([[0.0, 0.0]]), self.vertices, labels, outside_margin=self.outside_margin
        )
        loss_centroid = vertex_reconstruction_loss(
            torch.tensor([[5.1, 5.0]]), self.vertices, labels, outside_margin=self.outside_margin
        )
        assert loss_centroid.item() < loss_vertex.item()

    def test_loss_decreases_as_inside_point_approaches_centroid(self):
        """Moving an inside-labeled point closer to the hull center should reduce loss."""
        labels = torch.tensor([1])
        loss_far = vertex_reconstruction_loss(
            torch.tensor([[90, 90]]), self.vertices, labels, outside_margin=self.outside_margin
        )
        loss_close = vertex_reconstruction_loss(
            torch.tensor([[5.1, 5.0]]), self.vertices, labels, outside_margin=self.outside_margin   
        )
        assert loss_close.item() < loss_far.item()

    def test_loss_decreases_as_outside_point_moves_outward(self):
        """Moving an outside-labeled point further away should reduce loss (beyond margin)."""
        labels = torch.tensor([0])
        loss_near = vertex_reconstruction_loss(
            torch.tensor([[0.0, 0.0]]), self.vertices, labels, outside_margin=self.outside_margin
        )
        loss_far = vertex_reconstruction_loss(
            torch.tensor([[50.0, 50.0]]), self.vertices, labels, outside_margin=self.outside_margin
        )
        assert loss_far.item() < loss_near.item()

    def test_temperature_effect(self):
        """Lower temperature should sharpen weights toward nearest vertex."""
        outputs = torch.tensor([[0.1, 0.1]])
        labels = torch.tensor([1])
        loss_sharp = vertex_reconstruction_loss(outputs, self.vertices, labels, temperature=0.01, outside_margin=self.outside_margin)
        loss_smooth = vertex_reconstruction_loss(outputs, self.vertices, labels, temperature=10.0, outside_margin=self.outside_margin)
        assert loss_sharp.item() != pytest.approx(loss_smooth.item(), abs=1e-4)

