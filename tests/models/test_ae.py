import pytest
import torch

from torchstc.models import AutoEncoder

class TestAutoEncoder:
	def _create_autoencoder_model(self, 
								  hidden_units: list[int]):
		model = AutoEncoder(hidden_units)
		return model

	def test_model(self):
		hidden_units = [784, 256, 64, 32]
		model = self._create_autoencoder_model(hidden_units)
		# create random input and compare input and output shapes: laters should be equal
		x = torch.rand(32, 784)
		xx = model(x)
		assert xx.shape == x.shape
		# check if forward method works
		xx = model.forward(x)
		assert xx.shape == x.shape