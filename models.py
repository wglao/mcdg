from functools import partial
from typing import Callable, Iterable, NamedTuple, Sequence, Optional

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as tree
import jraph
import numpy as np
from jax import vmap
from jax import lax
from jax.lax import dynamic_slice_in_dim as dyslice


class DGMeshGraph(NamedTuple):
  """Tuple of graphs representing a DG discretization.
    Each interpolant node has an edge directed to the element node(s)
    it influences. Each element node has an edge directed to the
    interpolant nodes within its domain.
  """
  k: int
  n_p: int
  # edges: jnp.ndarray
  elements: jnp.ndarray
  interpolants: jnp.ndarray
  data: jnp.ndarray


def getGraph(data, k, n_p) -> DGMeshGraph:
  # Save the number of nodes and the number of edges.
  n_n = n_p + 1
  n_nodes = k*n_n
  # nodes contain an index used for slicing data
  nodes = jnp.arange(n_nodes)
  elements = jnp.arange(0, k*n_n, n_n)
  interpolants = vmap(
      lambda u, k, n: dyslice(u,
                              k*(n_n) + 1, n_p),
      in_axes=(None, 0, None))(nodes, jnp.arange(k), n_p)

  # We define edges with `senders` (source node indices) and `receivers`
  senders = []
  receivers = []
  # for node in nodes:
  #   elm, intrp = jnp.divmod(node, (n_n))

  #   if intrp == 0:
  #     # if element node
  #     temp = [node]*n_p
  #     senders += temp
  #     receivers += [temp[i] + i + 1 for i in range(n_p)]
  #   else:
  #     # if interpolant node
  #     senders += [node]
  #     receivers += [elm*(n_n)]
  #     if intrp == n_p:
  #       # if +boundary node
  #       senders += [node]
  #       if elm < k - 1:
  #         receivers += [(elm+1)*(n_n)]
  #       else:
  #         receivers += [0]

  # senders = jnp.array(senders)
  # receivers = jnp.array(receivers)
  # edges = jnp.stack((senders, receivers))

  return DGMeshGraph(k, n_p, elements, interpolants, data)


class ResNetBlock(nn.Module):
  """ResNet for update funcitons"""
  feature_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  out_sz: Optional[int] = None

  @nn.compact
  def __call__(self, u_n):
    if self.out_sz is None:
      out_sz = u_n.size
    else:
      out_sz = self.out_sz
    f = u_n

    for size in self.feature_sizes:
      # Residual Block
      f = nn.Dense(features=size)(f)
      f = self.activation(f)
      f = nn.Dense(features=size)(f)

      # output
      # f = self.activation(f)

    # return output to size of input
    f = nn.Dense(features=out_sz)(f)
    u_n_plus_1 = u_n + f
    return u_n_plus_1


class ElementGNN(nn.Module):
  """Graph Convolutional Network Operating on the DG Mesh Graph, Element-Wise"""
  latent_size: int
  num_resnet_blocks: int
  message_passing_steps: int
  k: int
  n_p: int

  def setup(self):
    resnet_feature_sizes = [self.latent_size]*self.num_resnet_blocks + [
        self.n_p
    ]
    self.resnet = ResNetBlock(resnet_feature_sizes)

  def updateElement(self, element, data):
    # concatenate all interpolant values sending
    # to the element
    data = jnp.asarray(jnp.reshape(data, (self.k, self.n_p)))
    elm_slice = jnp.where(
        element == 0,
        jnp.concatenate((jnp.array([self.k - 1]), jnp.zeros(
            (self.n_p,)), jnp.ones(1,)), None),
        jnp.where(
            element / (self.n_p + 1) == self.k - 1,
            jnp.concatenate((jnp.array([self.k - 2]), (self.k - 1)*jnp.ones(
                (self.n_p,)), jnp.zeros(1,)), None),
            jnp.concatenate(
                (jnp.array([element / (self.n_p + 1) - 1]), element /
                 (self.n_p + 1)*jnp.ones(
                     (self.n_p,)), jnp.array([element / (self.n_p + 1) + 1])),
                None)))
    intrp_slice = jnp.concatenate(
        (jnp.array([self.n_p - 1]), jnp.arange(self.n_p), jnp.zeros(
            (1,))), None)
    sent_features = data[elm_slice.astype(int), intrp_slice.astype(int)]
    element_state = self.resnet(sent_features)
    return element_state[1:-1]

  @nn.compact
  def __call__(self, mesh: DGMeshGraph) -> DGMeshGraph:
    # Apply a Graph Network once for each message-passing round.
    for _ in range(self.message_passing_steps):
      # update_edge_fn = None
      data = jnp.concatenate(
          vmap(self.updateElement, in_axes=(0, None))(mesh.elements, mesh.data),
          None)

    return DGMeshGraph(mesh.k, mesh.n_p, mesh.elements,
                       mesh.interpolants, data)
