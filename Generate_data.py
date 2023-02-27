import numpy as np
import jax
from jax import vmap, jit, random, lax
import pandas as pd
import jax.numpy as jnp

# from jax.config import config
# config.update("jax_enable_x64", True)

train_seed = random.PRNGKey(0)
test_seed = random.PRNGKey(1)

# ### Data Generation Inputs
num_train = 200
num_test = 10
nt_step_train = 41  # including the initial condition
nt_step_test = 401

# num_train = 2
# num_test = 2
# nt_step_train = 41 # including the initial condition
# nt_step_test = 31


def load_data(filename):
  np.loadtxt(filename, delimiter=',')


LIFT = np.loadtxt('MATLAB/LIFT.txt', delimiter=',')
Dr = np.loadtxt('MATLAB/Dr.txt', delimiter=',')
Fscale = np.loadtxt('MATLAB/Fscale.txt', delimiter=',')
rk4a = np.loadtxt('MATLAB/rk4a.txt', delimiter=',')
rk4b = np.loadtxt('MATLAB/rk4b.txt', delimiter=',')
rk4c = np.loadtxt('MATLAB/rk4c.txt', delimiter=',')
rx = np.loadtxt('MATLAB/rx.txt', delimiter=',')
vmapM = np.loadtxt('MATLAB/vmapM.txt', delimiter=',', dtype=int) - 1
vmapP = np.loadtxt('MATLAB/vmapP.txt', delimiter=',', dtype=int) - 1
vmapI = np.loadtxt('MATLAB/vmapI.txt', delimiter=',', dtype=int) - 1
vmapO = np.loadtxt('MATLAB/vmapO.txt', delimiter=',', dtype=int) - 1
mapI = np.loadtxt('MATLAB/mapI.txt', delimiter=',', dtype=int) - 1
mapO = np.loadtxt('MATLAB/mapO.txt', delimiter=',', dtype=int) - 1
x = np.loadtxt('MATLAB/x.txt', delimiter=',')
N = int(np.loadtxt('MATLAB/N.txt', delimiter=','))
K = int(np.loadtxt('MATLAB/K.txt', delimiter=','))
nx = (np.loadtxt('MATLAB/nx.txt', delimiter=','))

EToE = (np.loadtxt('MATLAB/EToE.txt', delimiter=',') - 1).astype(int)

Np = N + 1
Nfp = 1
Nfaces = 2
a = 1
alpha = 1

modes = 3
Basis = np.zeros((modes, Np, K))
# for i in range(1, int(modes/2) + 1):
#     Basis[2*i-2, :] = np.sin(np.pi * 2 * i * x)
#     Basis[2*i-1, :] = np.cos(np.pi * 2 * i * x)

# Basis = jnp.asarray(Basis)

FinalTime = 0.4

# compute time step size
xmin = np.min(np.abs(x[1, :] - x[2, :]))
CFL = 0.75
dt = CFL / (1)*xmin
dt = .5*dt

print(dt)

Nsteps = int(np.ceil(FinalTime / dt))
dt = FinalTime / Nsteps
dt = np.round(dt*100) / 100

dt = 0.01

print(dt, Nsteps)


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


def numerical_solver(u):
  resu = jnp.zeros((Np, K))
  for INTRK in range(0, 5):
    rhsu = AdvecRHS1D(u)
    resu = rk4a[INTRK]*resu + dt*rhsu
    u = u + rk4b[INTRK]*resu
  return u


def generate_data(u, Nsteps):
  step_save = 0
  U_data = jnp.zeros((Nsteps, K*(N+1)))
  U_data = U_data.at[step_save, :].set((u.T).flatten())

  for tstep in range(1, Nsteps):  # outer time step loop
    u = numerical_solver(u)
    step_save += 1
    U_data = U_data.at[step_save, :].set((u.T).flatten())

  return U_data


generate_data_batch = vmap(generate_data, in_axes=(0, None))


def generate_delta(x, coefs):
  coefs = np.sort(np.abs(coefs))
  x1, umax, x2 = coefs
  umax = 5*umax
  xpeak = (x1+x2) / 2

  u = np.where(x <= x1, np.ones_like(x), np.where(x >= x2), np.ones_like(x),
               umax - 2*(umax-1)*(np.abs(x - xpeak) / (x2-x1)))
  
  return u


### Generate train data
print('Generating train data ......................')
coeffs_train = random.normal(train_seed, (num_train, modes))
# u_batch_train = np.einsum('bi, ikl -> bkl', coeffs_train, Basis)
u_batch_train = vmap(generate_delta, in_axes=(None, 0))(x, coeffs_train)
train_data = generate_data_batch(u_batch_train, nt_step_train)
print(train_data.shape)
print(train_data.max())
Solution_samples_array = pd.DataFrame({'samples': train_data.flatten()})
Solution_samples_array.to_csv(
    'data/Train_noise_' + str(0.00) + '_d_' + str(num_train) + '_Nt_' +
    str(nt_step_train) + '_K_' + str(K) + '_Np_' + str(N) + '.csv',
    index=False)

### Generate test data
print('Generating test data ......................')
coeffs_test = random.normal(test_seed, (num_test, modes))
# u_batch_test = np.einsum('bi, ikl -> bkl', coeffs_test, Basis)
u_batch_test = vmap(generate_delta, in_axes=(None, 0))(x, coeffs_test)
test_data = generate_data_batch(u_batch_test, nt_step_test)
print(test_data.shape)
print(test_data.max())
Solution_samples_array = pd.DataFrame({'samples': test_data.flatten()})
Solution_samples_array.to_csv(
    'data/Test_d_' + str(num_test) + '_Nt_' + str(nt_step_test) + '_K_' +
    str(K) + '_Np_' + str(N) + '.csv',
    index=False)

### Generate test data
print('Generating test data ......................')
nt_test = 401*5
test_data = generate_data_batch(u_batch_test, nt_test)
print(test_data.shape)
Solution_samples_array = pd.DataFrame({'samples': test_data.flatten()})
Solution_samples_array.to_csv(
    'data/Test_d_' + str(num_test) + '_Nt_' + str(nt_test) + '_K_' + str(K) +
    '_Np_' + str(N) + '.csv',
    index=False)

# print('='*20 + ' TRAIN NOISE DATA () ' + '='*20)
key_data_noise = random.PRNGKey(3)
U_train = np.reshape(train_data, (num_train, nt_step_train, K, N + 1))
nosie_vec = jax.random.normal(key_data_noise, U_train.shape)

noise_level = 0.01


def element_adding_noise(element_nodal_values, nosie_vec):
  return element_nodal_values + noise_level*nosie_vec*jnp.max(
      element_nodal_values)


batch_element_adding_noise = jax.vmap(element_adding_noise, in_axes=(0, 0))


def timestep_adding_noise(step_all_element_values, nosie_vec):
  return batch_element_adding_noise(step_all_element_values, nosie_vec)


batch_timestep_adding_noise = jit(
    jax.vmap(timestep_adding_noise, in_axes=(0, 0)))


def sample_adding_noise(sample_values, nosie_vec):
  return batch_timestep_adding_noise(sample_values, nosie_vec)


batch_sample_adding_noise = (jax.vmap(sample_adding_noise, in_axes=(0, 0)))

train_data_noise = batch_sample_adding_noise(U_train, nosie_vec)

train_noise = pd.DataFrame({'samples': train_data_noise.flatten()})
train_noise.to_csv(
    'data/Train_noise_' + str(noise_level) + '_d_' + str(num_train) + '_Nt_' +
    str(nt_step_train) + '_K_' + str(K) + '_Np_' + str(N) + '.csv',
    index=False)

senders = []
receivers = []
for ele_index in range(K):
  for face_index in range(2):
    if ele_index != EToE[ele_index, face_index]:
      senders.append(ele_index)
      receivers.append(EToE[ele_index, face_index])

    if ele_index == EToE[ele_index, face_index]:
      senders.append(ele_index)
      receivers.append(ele_index)

senders_data = jnp.asarray(senders)
receivers_data = jnp.asarray(receivers)

# Imposing the periodic boundary conditions
receivers_data = receivers_data.at[0].set(19)
receivers_data = receivers_data.at[-1].set(0)

Solution_samples_array = pd.DataFrame(
    {'samples': np.asarray([len(senders_data)])})
Solution_samples_array.to_csv('MATLAB/num_edges.csv', index=False)

Solution_samples_array = pd.DataFrame({'samples': senders_data.flatten()})
Solution_samples_array.to_csv('MATLAB/senders_data.csv', index=False)

Solution_samples_array = pd.DataFrame({'samples': receivers_data.flatten()})
Solution_samples_array.to_csv('MATLAB/receivers_data.csv', index=False)

Edges_to_nodes = jnp.reshape(jnp.asarray(range(len(senders_data))), (-1, 2))
np.savetxt("MATLAB/edges_to_nodes.txt", Edges_to_nodes, delimiter=',')
