# ! This part is for supercomputing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node", default=1, type=int)
parser.add_argument("--GPU_index", default=0, type=int)
parser.add_argument("--alpha1", default=100, type=float)
parser.add_argument("--alpha2", default=2, type=float)
parser.add_argument("--alpha3", default=512, type=int)
parser.add_argument("--alpha4", default=256, type=float)
parser.add_argument("--alpha5", default=0.0, type=float)
parser.add_argument("--alpha6", default=0.0, type=float)
args = parser.parse_args()

NODE = args.node
GPU_index = args.GPU_index

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_index)

import pickle
import time
from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as tree
import jraph
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from jax import grad, jit, lax, vmap
from jraph._src import utils

import models

# int(args.alpha4) blank
# ## Loading data

n_seq = 0
mc_alpha = 0

num_train = 400
num_test = 10

K = int(args.alpha1)
N = int(args.alpha2)

nt_train_arr = jnp.array([[11, 27, 54], [31, 78, 155], [107, 267, 533]])
nt_step_train = int(nt_train_arr[jnp.where(N == jnp.array([2, 4, 8])),
                                 jnp.where(K == jnp.array([20, 50, 100]))][0,
                                                                           0])
nt_test_arr = jnp.array([[108, 268, 535], [310, 774, 1546], [1066, 2662, 5323]])
nt_step_test = int(nt_test_arr[jnp.where(N == jnp.array([2, 4, 8])),
                               jnp.where(K == jnp.array([20, 50, 100]))][0, 0])

learning_rate = 1e-4
num_epochs = 10000

wandb_upload = True

batch_size = num_train if num_train == 2 else 10
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = int(num_complete_batches) + int(bool(leftover))

# num_train = 2
# num_test = 2
# nt_step_train = 41
# nt_step_test = 31
# num_epochs = 10
# wandb_upload = False

proj = partial(os.path.join, os.environ["PROJ"])

# In time space
LIFT = jnp.array(
    np.loadtxt(
        proj('MATLAB/LIFT_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
Dr = jnp.array(
    np.loadtxt(
        proj('MATLAB/Dr_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
Fscale = jnp.array(
    np.loadtxt(
        proj('MATLAB/Fscale_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
invV = jnp.array(
    np.loadtxt(
        proj('MATLAB/invV_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
rk4a = jnp.array(np.loadtxt(proj('MATLAB/rk4a.txt'), delimiter=','))
rk4b = jnp.array(np.loadtxt(proj('MATLAB/rk4b.txt'), delimiter=','))
rk4c = jnp.array(np.loadtxt(proj('MATLAB/rk4c.txt'), delimiter=','))
rx = jnp.array(
    np.loadtxt(
        proj('MATLAB/rx_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
V = jnp.array(
    np.loadtxt(
        proj('MATLAB/V_K_' + str(K) + '_Np_' + str(N) + '.txt'), delimiter=','))
vmapM = jnp.array(
    np.loadtxt(
        proj('MATLAB/vmapM_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
vmapP = jnp.array(
    np.loadtxt(
        proj('MATLAB/vmapP_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
vmapI = jnp.array(
    np.loadtxt(
        proj('MATLAB/vmapI_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
vmapO = jnp.array(
    np.loadtxt(
        proj('MATLAB/vmapO_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
mapI = jnp.array(
    np.loadtxt(
        proj('MATLAB/mapI_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
mapO = jnp.array(
    np.loadtxt(
        proj('MATLAB/mapO_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',',
        dtype=int) - 1)
x = jnp.array(
    np.loadtxt(
        proj('MATLAB/x_K_' + str(K) + '_Np_' + str(N) + '.txt'), delimiter=','))
nx = jnp.array(
    np.loadtxt(
        proj('MATLAB/nx_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=','))
EToE = jnp.array(
    np.loadtxt(
        proj('MATLAB/EToE_K_' + str(K) + '_Np_' + str(N) + '.txt'),
        delimiter=',') - 1).astype(int)

Np = N + 1
Nfp = 1
Nfaces = 2
a = 1
alpha = 1

dt = 1 / (nt_step_test-1)

noise_level = args.alpha6

filename = 'Adv_plateau_DG_GNN_K_' + str(K) + '_Np_' + str(
    N) + '_element_graph_MCalpha_' + str(
        mc_alpha) + '_noise_' + '{:.2f}'.format(noise_level) + '_lr_' + str(
            learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(
                n_seq) + '_num_epochs_' + str(num_epochs)

if wandb_upload:
  import wandb
  wandb.init(project="DG_GNN_mcTangent_Approach", entity="wglao", name=filename)
  wandb.config.problem = 'Advection/plateau'
  wandb.config.method = 'element graph'
  wandb.config.batchsize = str(batch_size)
  wandb.config.Seq_ML = str(n_seq)
  wandb.config.database = str(num_train)
  wandb.config.noise_level = '{:.2f}'.format(noise_level)

#! 1. Loading data by pandas
print('='*20 + ' >>')
print('Loading train data ...')
Train_data = pd.read_csv(
    proj('data/plateau/Train_noise_') + str(noise_level) + '_d_' +
    str(num_train) + '_Nt_' + str(nt_step_train) + '_K_' + str(K) + '_Np_' +
    str(N) + '.csv')
Train_data = np.reshape(Train_data.to_numpy(), (num_train, nt_step_train, K*Np))

print(Train_data.shape)
print('='*20 + ' >>')
print('Loading test data ...')

Test_data = pd.read_csv(
    proj('data/plateau/Test_d_') + str(num_test) + '_Nt_' + str(nt_step_test) +
    '_K_' + str(K) + '_Np_' + str(N) + '.csv')
Test_data = np.reshape(Test_data.to_numpy(), (num_test, nt_step_test, K*Np))

print(Test_data.shape)

#! 2. Building up a neural network
graph_base = models.getGraph(Train_data[0, 0, :], K, Np)
net = models.ElementGNN(100, 3, 1, K, Np)
init_params = net.init(jax.random.PRNGKey(1), graph_base)

optimizer = optax.adam(learning_rate)
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
                                       (1-alpha)*jnp.abs(nx_transpose)) / 2

  # Impose periodic conditions
  # impose boundary condition at x=0
  uin = u_transpose[-1]
  du_transpose = du_transpose.at[mapI].set(
      (u_transpose[vmapI] - uin)*(nx_transpose[mapI] -
                                  (1-alpha)*jnp.abs(nx_transpose[mapI])) / 2)

  # impose boundary condition at x=1
  uout = u_transpose[0]
  du_transpose = du_transpose.at[mapO].set(
      (uout - u_transpose[vmapO])*(nx_transpose[mapI] -
                                   (1-alpha)*jnp.abs(nx_transpose[mapI])) / 2)

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


dt_factor = 1


def body_func(carriers, noise):
  params, u_ti = carriers
  u_temp = u_ti + noise
  return carriers, u_ti - dt_factor*dt*net.apply(params, u_temp)


def single_forward_pass(params, u_ti):
  # _, u_sets = lax.scan(
  #     body_func, (params, u_ti),
  #     noise_level*jrand.generalized_normal(jrand.PRNGKey(1),(u_ti.shape)))
  # return jnp.mean(u_sets, axis=0)
  g = np.random.default_rng()
  u_noise = u_ti + u_ti*noise_level*g.normal(size=u_ti.shape)
  mesh = models.getGraph(u_noise, K, Np)
  mesh = net.apply(params, mesh)
  return u_ti - dt_factor*dt*mesh.data


#! 4. Training loss functions
def MSE(pred, true):
  return jnp.mean(jnp.square(pred - true))


def squential_S_phase(i, args):

  loss_ml, loss_mc, u_tilde, u_true, params = args

  # The numerical solver solutions u_tilde branch
  # u_bar = single_solve_forward(u_tilde)

  # The neural network solver solutions u_tilde branch
  u_tilde = single_forward_pass(params, u_tilde)

  # The machine learning loss
  loss_ml += MSE(u_tilde, u_true[i + 1])

  # The model-constrained loss
  # loss_mc += MSE(u_tilde, u_bar)

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


@jit
def LossmcDNN(params, data):
  return jnp.sum(loss_one_sample_batch(params, data))


# ## Computing test error, predictions over all time steps
def solve_body(i, args):
  params, u_data_current = args
  u_next = single_forward_pass(params, u_data_current[i - 1, ...])
  u_data_current = u_data_current.at[i, :].set(u_next)
  return params, u_data_current


def neural_solver(params, U_test):
  u = U_test[0, :]
  U = jnp.zeros((nt_step_test, K*int(Np)))
  U = U.at[0, :].set(u)
  _, U = lax.fori_loop(1, nt_step_test, solve_body, (params, U))
  return U


neural_solver_batch = vmap(neural_solver, in_axes=(None, 0))


@jit
def test_acc(params, Test_set):
  return MSE(neural_solver_batch(params, Test_set), Test_set)


#! 5. Epoch loop and training settings
@jit
def body_fun(i, args):
  params, opt_state, data = args
  data_batch = lax.dynamic_slice_in_dim(data, i*batch_size, batch_size)
  gradients = grad(LossmcDNN)(params, data_batch)
  updates, opt_state = optimizer.update(gradients, opt_state)
  params = optax.apply_updates(params, updates)
  return (params, opt_state, data)


def run_epoch(params, opt_state, data):
  params, opt_state, _ = lax.fori_loop(0, num_batches, body_fun,
                                       (params, opt_state, data))
  return params


def TrainModel(train_data, test_data, num_epochs, params, opt_state,
               wandb_upload):
  plot_sample = 3

  test_accuracy_min = 100
  epoch_min = -1

  best_params = params

  for epoch in range(num_epochs):
    Begin_time_1 = time.time()
    params = run_epoch(params, opt_state, train_data)
    End_time_1 = time.time()
    Epoch_training_1 = End_time_1 - Begin_time_1

    train_loss = LossmcDNN(params, train_data)
    test_accuracy = test_acc(params, test_data)

    if epoch == 0:
      best_f = plt.figure()
      for i in range(K):
        plt.plot(
            x[:, i],
            np.reshape(Test_data[plot_sample, 0, :], (K, N + 1))[i, :],
            linestyle='-',
            color='k',
            dashes=(2, 2))

    if test_accuracy_min >= test_accuracy:
      test_accuracy_min = test_accuracy
      epoch_min = epoch
      # trained_params = optimizers.unpack_optimizer_state(optimal_opt_state)
      pf = open(proj('Network/Best_') + filename, 'wb')
      pickle.dump(params, pf)
      pf.close()

      plt.close('all')
      best_f = plt.figure()
      pred = neural_solver(params, Test_data[plot_sample, :, :])
      for i in range(K):
        plt.plot(
            x[:, i],
            np.reshape(Test_data[plot_sample, 0, :], (K, N + 1))[i, :],
            linestyle='-',
            color='k',
            dashes=(2, 2))
        plt.plot(
            x[:, i],
            np.reshape(pred[int(3*nt_step_test / 4), :], (K, N + 1))[i, :],
            linestyle='-',
            color='b',
            marker='o',
            markevery=N,
            ms=5,
            mfc='w')
        plt.plot(
            x[:, i],
            np.reshape(Test_data[plot_sample,
                                 int(3*nt_step_test / 4), :], (K, N + 1))[i, :],
            linestyle='-',
            color='r',
            marker='x',
            dashes=(3, 3),
            markevery=3,
            ms=5)

    if (not wandb_upload) & (epoch % 1 == 0):  # Print MSE every 100 epochs
      print(
          "Data_d {:d} n_seq {:d} batch {:d} lr {:.2e} loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} Time {:.3e}s"
          .format(num_train, n_seq, batch_size, learning_rate, train_loss,
                  test_accuracy, test_accuracy_min, epoch_min, epoch,
                  Epoch_training_1))

    if (wandb_upload) & (epoch % 100 == 0):  # Print MSE every 100 epochs
      test_f = plt.figure()
      pred = neural_solver(params, Test_data[plot_sample, :, :])
      for i in range(K):
        plt.plot(
            x[:, i],
            np.reshape(Test_data[plot_sample, 0, :], (K, N + 1))[i, :],
            linestyle='-',
            color='k',
            dashes=(2, 2))
        plt.plot(
            x[:, i],
            np.reshape(pred[int(3*nt_step_test / 4), :], (K, N + 1))[i, :],
            linestyle='-',
            color='b',
            marker='o',
            markevery=N,
            ms=5,
            mfc='w')
        plt.plot(
            x[:, i],
            np.reshape(Test_data[plot_sample,
                                 int(3*nt_step_test / 4), :], (K, N + 1))[i, :],
            linestyle='-',
            color='r',
            marker='x',
            dashes=(3, 3),
            markevery=3,
            ms=5)
      wandb.log({
          "Train loss": float(train_loss),
          "Test Error": float(test_accuracy),
          "TEST MIN": float(test_accuracy_min),
          "Test Plot": test_f,
          "Best Plot": best_f,
      })
      plt.close(test_f)

  return best_params, params


#! STEP6:: Training
# pf = open(
#     proj(
#         'Network/Best_Adv_plateau_DG_GNN_K_100_Np_2_flux_dim512-256MCalpha_0_noise_0.0_lr_0.001_batch_10_nseq_1_num_epochs_7500'
#     ), 'rb')
# init_params = pickle.load(pf)
# pf.close()
best_params, end_params = TrainModel(Train_data, Test_data, num_epochs,
                                     init_params, opt_state, wandb_upload)
# optimum_params = opt_get_params(best_opt_state)

# trained_params = optimizers.unpack_optimizer_state(best_opt_state)
# pickle.dump(trained_params, open(proj('Network/Best_' + filename, "wb"))

# trained_params = optimizers.unpack_optimizer_state(end_opt_state)
# pickle.dump(trained_params, open(proj('Network/End_' + filename, "wb"))

# best_params = pickle.load(open('Network/Best_' + filename, "rb"))
# opt_state = optimizers.pack_optimizer_state(best_params)
