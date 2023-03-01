# ! This part is for supercomputing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node", default=1, type=int)
parser.add_argument("--GPU_index", default=0, type=int)
parser.add_argument("--alpha1", default=0., type=float)
parser.add_argument("--alpha2", default=0.0, type=float)
parser.add_argument("--alpha3", default=10, type=int)
parser.add_argument("--alpha4", default=0., type=float)
parser.add_argument("--alpha5", default=0., type=float)
args = parser.parse_args()

NODE = args.node
GPU_index = args.GPU_index

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_index)

from functools import partial

import jax
# from jax.example_libraries import optimizers
import jax.numpy as jnp
from jax import grad, vmap, jit, lax
import jraph
import haiku as hk
import jax.tree_util as tree
from jraph._src import utils

import optax

import pandas as pd
import numpy as np

import pickle
import time

# int(args.alpha4) blank
# ## Loading data

n_seq = 1
mc_alpha = int(args.alpha1)

num_train = 200
num_test = 10
nt_step_train = 41
nt_step_test = 401

learning_rate = 1e-3
num_epochs = 1000000

wandb_upload = bool(int(args.alpha4))
adam_flag = bool(int(args.alpha5))

batch_size = num_train if num_train == 2 else 10
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = int(num_complete_batches) + int(bool(leftover))

# num_train = 2
# num_test = 2
# nt_step_train = 41
# nt_step_test = 31
# num_epochs = 10
# wandb_upload = False

proj = partial(os.path.join,os.environ["PROJ"])

# In time space
N = int(np.loadtxt(proj('MATLAB/N.txt'), delimiter=','))
K = int(np.loadtxt(proj('MATLAB/K.txt'), delimiter=','))
x = np.loadtxt(proj('MATLAB/x.txt'), delimiter=',')
LIFT = np.loadtxt(proj('MATLAB/LIFT.txt'), delimiter=',')
Dr = np.loadtxt(proj('MATLAB/Dr.txt'), delimiter=',')
Fscale = np.loadtxt(proj('MATLAB/Fscale.txt'), delimiter=',')
rk4a = np.loadtxt(proj('MATLAB/rk4a.txt'), delimiter=',')
rk4b = np.loadtxt(proj('MATLAB/rk4b.txt'), delimiter=',')
rx = np.loadtxt(proj('MATLAB/rx.txt'), delimiter=',')
vmapM = np.loadtxt(proj('MATLAB/vmapM.txt'), delimiter=',', dtype=int) - 1
vmapP = np.loadtxt(proj('MATLAB/vmapP.txt'), delimiter=',', dtype=int) - 1
vmapI = np.loadtxt(proj('MATLAB/vmapI.txt'), delimiter=',', dtype=int) - 1
vmapO = np.loadtxt(proj('MATLAB/vmapO.txt'), delimiter=',', dtype=int) - 1
mapI = np.loadtxt(proj('MATLAB/mapI.txt'), delimiter=',', dtype=int) - 1
mapO = np.loadtxt(proj('MATLAB/mapO.txt'), delimiter=',', dtype=int) - 1
nx = np.loadtxt(proj('MATLAB/nx.txt'), delimiter=',')
Np = N + 1
Nfp = 1
Nfaces = 2
a = 1
alpha = 1

dt = 0.01

noise_level = args.alpha2

filename = 'Adv_2delta_DG_GNN_NO_EMBED_receiver_flux_dim' + str(
    int(args.alpha3)) + 'MCalpha_' + str(mc_alpha) + '_noise_' + str(
        noise_level) + '_lr_' + str(learning_rate) + '_batch_' + str(
            batch_size) + '_nseq_' + str(n_seq) + '_num_epochs_' + str(
                num_epochs)

if wandb_upload:
  import wandb
  wandb.init(project="DG_GNN_mcTangent_Approach", entity="wglao", name=filename)
  wandb.config.problem = 'Advection/2delta'
  wandb.config.method = 'receivers'
  wandb.config.batchsize = str(batch_size)
  wandb.config.Seq_ML = str(n_seq)
  wandb.config.database = str(num_train)
  wandb.config.noise_level = str(noise_level)

#! 1. Loading data by pandas
print('='*20 + ' >>')
print('Loading train data ...')
Train_data = pd.read_csv(proj('data/2delta/Train_noise_') + str(noise_level) + '_d_' +
                         str(num_train) + '_Nt_' + str(nt_step_train) + '_K_' +
                         str(K) + '_Np_' + str(N) + '.csv')
Train_data = np.reshape(Train_data.to_numpy(), (num_train, nt_step_train, K*Np))

print(Train_data.shape)
print('='*20 + ' >>')
print('Loading test data ...')

Test_data = pd.read_csv(proj('data/2delta/Test_d_') + str(num_test) + '_Nt_' +
                        str(nt_step_test) + '_K_' + str(K) + '_Np_' + str(N) +
                        '.csv')
Test_data = np.reshape(Test_data.to_numpy(), (num_test, nt_step_test, K*Np))

print(Test_data.shape)


#! 2. Building up a neural network
def build_toy_graph(u) -> jraph.GraphsTuple:
  """Define a four node graph, each node has a scalar as its feature."""

  # Nodes are defined implicitly by their features.
  # We will add four nodes, each with a feature, e.g. each node has feature of N_p values,
  node_features = jnp.reshape(u, (K, Np))

  # We will now specify 5 directed edges connecting the nodes we defined above.
  # We define this with `senders` (source node indices) and `receivers`
  senders = jnp.concatenate((jnp.asarray(range(K)), jnp.asarray(range(K))))
  receivers = jnp.concatenate(
      (jnp.roll(jnp.asarray(range(K)), -1), jnp.roll(jnp.asarray(range(K)), 1)))

  temp = receivers
  receivers = senders
  senders = temp

  # You can optionally add edge attributes to the 5 edges.
  # WE DO NOT NEED EGED ATTRIBUTES, AT THE MOMENT
  edges = jnp.array([[0.]])

  # We then save the number of nodes and the number of edges.
  n_node = jnp.array([K])
  n_edge = jnp.array([2*K])

  # Optionally you can add `global` information, such as a graph label.
  global_context = jnp.array([[1]])  # Same feature dims as nodes and edges.
  graph = jraph.GraphsTuple(
      nodes=node_features,
      edges=edges,
      senders=senders,
      receivers=receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=global_context)
  return graph


graph_base = build_toy_graph(Train_data[0, 0, :])


def u_2_graph(u):
  """Assign new node attributes to the base graph"""
  A = jnp.reshape(u, (K, Np))
  B = jnp.concatenate((A, x.T), axis=1)
  return graph_base._replace(nodes=B)


def GraphMapFeatures(embed_node_fn):

  def Embed(graphs_tuple):
    return graphs_tuple._replace(nodes=embed_node_fn(graphs_tuple.nodes))

  return Embed


def net_embed(graphs_tuple):
  embedder = GraphMapFeatures(
      embed_node_fn=hk.Sequential([
          hk.Linear(int(args.alpha3)), jax.nn.relu,
          hk.Linear(int(args.alpha3))
      ]))
  return embedder(graphs_tuple)


def GraphNetwork_simplified(update_edge_fn,
                            update_node_fn,
                            aggregate_edges_for_nodes_fn=utils.segment_sum):

  def _ApplyGraphNet(graph):

    nodes, _, receivers, senders, _, _, _ = graph

    sum_n_node = tree.tree_leaves(nodes)[0].shape[0]

    # Paring two elements of which the egde connects
    edges = update_edge_fn(
        tree.tree_map(lambda n: n[senders], nodes),
        tree.tree_map(lambda n: n[receivers], nodes))

    # Updating the nodes from neighbor egdes and the element/node itself
    received_neighbor_attributes = jnp.concatenate((edges[K:], edges[:K]),
                                                   axis=1)
    nodes = update_node_fn(nodes, received_neighbor_attributes)

    return nodes.flatten(
    )  # we can return a graph, but we do not need the graph at the moment

  return _ApplyGraphNet


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(int(args.alpha3)), jax.nn.relu,
       hk.Linear(int(args.alpha3))])
  return net(feats)


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [hk.Linear(int(args.alpha3)), jax.nn.relu,
       hk.Linear(int(Np))])
  return net(feats)


def net_update(graphs_tuple):
  net = GraphNetwork_simplified(
      update_node_fn=node_update_fn, update_edge_fn=edge_update_fn)
  return net(graphs_tuple)


def net_fn(u):
  return net_update((u_2_graph(u)))


net = hk.without_apply_rng(hk.transform(net_fn))
init_params = net.init(jax.random.PRNGKey(1), Train_data[0, 0, :])

if adam_flag:
  optimizer = optax.adam(learning_rate)
else:
  optimizer = optax.eve(learning_rate)
opt_state = optimizer.init(init_params)

#! 3. Numerical solver + neural network solver (single time step)
#! STEP 3.1:: Define numerical forward solver (Back-Euler scheme)


def AdvecRHS1D(u):
  u_transpose = u.T.flatten()
  nx_transpose = nx.T.flatten()
  # form field differences at faces
  alpha = 1
  du_transpose = (u_transpose[vmapM] -
                  u_transpose[vmapP])*(nx_transpose -
                                       (1-alpha)*np.abs(nx_transpose)) / 2

  # Impose periodic conditions
  # impose boundary condition at x=0
  uin = u_transpose[-1]
  du_transpose = du_transpose.at[mapI].set(
      (u_transpose[vmapI] - uin)*(nx_transpose[mapI] -
                                  (1-alpha)*np.abs(nx_transpose[mapI])) / 2)

  # impose boundary condition at x=1
  uout = u_transpose[0]
  du_transpose = du_transpose.at[mapO].set(
      (uout - u_transpose[vmapO])*(nx_transpose[mapI] -
                                   (1-alpha)*np.abs(nx_transpose[mapI])) / 2)

  # compute right hand sides of the semi-discrete PDE
  du = jnp.reshape(du_transpose, (K, Nfp*Nfaces)).T
  rhsu = -rx*(Dr@u) + LIFT @ (Fscale*(du))

  return rhsu


def single_solve_forward(u_ti):
  u = jnp.reshape(u_ti, (K, Np)).T
  resu = jnp.zeros((Np, K))
  for INTRK in range(0, 5):
    rhsu = AdvecRHS1D(u)
    resu = rk4a[INTRK]*resu + dt*rhsu
    u = u + rk4b[INTRK]*resu
  return u.T.flatten()


#! STEP 3.2:: Define neural network forward solver (Back-Euler scheme)
def single_forward_pass(params, u_ti):
  # u = single_solve_forward(u_ti) # This is for debugging the training architecture
  u = u_ti - dt*net.apply(params, u_ti)
  return u


#! 4. Training loss functions
def MSE(pred, true):
  return jnp.mean(jnp.square(pred - true))


def squential_S_phase(i, args):

  loss_ml, loss_mc, u_tilde, u_true, params = args

  # The numerical solver solutions u_tilde branch
  u_bar = single_solve_forward(u_tilde)

  # The neural network solver solutions u_tilde branch
  u_tilde = single_forward_pass(params, u_tilde)

  # The machine learning loss
  loss_ml += MSE(u_tilde, u_true[i + 1])

  # The model-constrained loss
  loss_mc += MSE(u_tilde, u_bar)

  return loss_ml, loss_mc, u_tilde, u_true, params


def loss_one_sample_one_time(params, u):
  loss_ml = 0
  loss_mc = 0
  u0_tilde = u[0, :]

  # for the following steps up to sequential steps n_seq
  loss_ml, loss_mc, _, _, _ = lax.fori_loop(
      0, n_seq + 1, squential_S_phase, (loss_ml, loss_mc, u0_tilde, u, params))

  return loss_ml + mc_alpha*loss_mc


loss_one_sample_one_time_batch = jit(
    vmap(loss_one_sample_one_time, in_axes=(None, 0), out_axes=0))


def transform_one_sample_data(u_one_sample):
  u_out = jnp.zeros((nt_step_train - n_seq - 1, n_seq + 2, K*Np))
  for i in range(nt_step_train - n_seq - 1):
    u_out = u_out.at[i, :, :].set(u_one_sample[i:i + n_seq + 2, :])
  return u_out


def loss_one_sample(params, u_one_sample):
  u_batch_nt = transform_one_sample_data(u_one_sample)
  return jnp.sum(loss_one_sample_one_time_batch(params, u_batch_nt))


loss_one_sample_batch = jit(
    vmap(loss_one_sample, in_axes=(None, 0), out_axes=0))


def LossmcDNN(params, data):
  return jnp.sum(loss_one_sample_batch(params, data))


# ## Computing test error, predictions over all time steps
def neural_solver(params, U_test):

  u = U_test[0, :]
  U = jnp.zeros((nt_step_test, K*Np))
  U = U.at[0, :].set(u)

  for i in range(1, nt_step_test):
    u = single_forward_pass(params, u)
    U = U.at[i, :].set(u)

  return U


neural_solver_batch = vmap(neural_solver, in_axes=(None, 0))


@jit
def test_acc(params, Test_set):
  return MSE(neural_solver_batch(params, Test_set), Test_set)


#! 5. Epoch loop and training settings
def body_fun(i, args):
  params, opt_state, data = args
  data_batch = lax.dynamic_slice_in_dim(data, i*batch_size, batch_size)
  gradients = grad(LossmcDNN)(params, data_batch)  
  updates, opt_state = optimizer.update(gradients, opt_state)
  params = optax.apply_updates(params, updates)
  return (params, opt_state, data)


def run_epoch(params, opt_state, data):
  params, opt_state, _ = lax.fori_loop(0, num_batches, body_fun, (params, opt_state, data))
  return params


def TrainModel(train_data, test_data, num_epochs, params, opt_state, wandb_upload):

  test_accuracy_min = 100
  epoch_min = -1

  best_params = params

  for epoch in range(num_epochs):
    Begin_time_1 = time.time()
    params = run_epoch(params, opt_state, train_data)
    End_time_1 = time.time()
    Epoch_training_1 = End_time_1 - Begin_time_1

    train_loss = LossmcDNN(params, train_data)
    if not adam_flag:
      opt_state.hyperparams['f'] = train_loss
    test_accuracy = test_acc(params, test_data)

    if test_accuracy_min >= test_accuracy:
      test_accuracy_min = test_accuracy
      epoch_min = epoch
      best_params = params

    if (not wandb_upload) & (epoch % 1 == 0):  # Print MSE every 100 epochs
      print(
          "Data_d {:d} n_seq {:d} batch {:d} lr {:.2e} loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} Time 1 {:.3e}s"
          .format(num_train, n_seq, batch_size, learning_rate, train_loss,
                  test_accuracy, test_accuracy_min, epoch_min, epoch,
                  Epoch_training_1))

    if (wandb_upload) & (epoch % 5000 == 0):  # Print MSE every 100 epochs
      wandb.log({
          "Train loss": float(train_loss),
          "Test Error": float(test_accuracy),
          'TEST MIN': float(test_accuracy_min)
      })
      # trained_params = optimizers.unpack_optimizer_state(optimal_opt_state)
      pickle.dump(best_params, open(proj('Network/Best_') + filename, "wb"))

      # trained_params = optimizers.unpack_optimizer_state(opt_state)
      pickle.dump(params, open(proj('Network/End_') + filename, "wb"))

  return best_params, params


#! STEP6:: Training
best_params, end_params = TrainModel(Train_data, Test_data, num_epochs,
                                     init_params, opt_state, wandb_upload)
# optimum_params = opt_get_params(best_opt_state)

# trained_params = optimizers.unpack_optimizer_state(best_opt_state)
# pickle.dump(trained_params, open(proj('Network/Best_' + filename, "wb"))

# trained_params = optimizers.unpack_optimizer_state(end_opt_state)
# pickle.dump(trained_params, open(proj('Network/End_' + filename, "wb"))

# best_params = pickle.load(open('Network/Best_' + filename, "rb"))
# opt_state = optimizers.pack_optimizer_state(best_params)
