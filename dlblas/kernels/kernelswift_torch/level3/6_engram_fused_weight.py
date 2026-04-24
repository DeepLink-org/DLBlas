import torch
import torch.nn as nn


class Model(nn.Module):
  def __init__(self, hc_mult: int, hidden_size: int):
      super().__init__()
      self.hc_mult = hc_mult
      self.hidden_size = hidden_size

  def forward(self, wh_data, we_data):
      return wh_data.float() * we_data.float()


hc_mult = 4
hidden_size = 128


def generate_test_data(hc_mult, hidden_size):
  wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
  we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16)
  return [wh_data, we_data]


def get_inputs():
  wh_data, we_data = generate_test_data(hc_mult, hidden_size)
  return [wh_data, we_data]


def get_init_inputs():
  return [hc_mult, hidden_size]