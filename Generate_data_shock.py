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

# num_train = 2
# num_test = 2
# nt_step_train = 41 # including the initial condition
# nt_step_test = 31


def load_data(filename):
  jnp.array(np.loadtxt(filename, delimiter=','))


LIFT = jnp.array(np.loadtxt('MATLAB/LIFT.txt', delimiter=','))
Dr = jnp.array(np.loadtxt('MATLAB/Dr.txt', delimiter=','))
Fscale = jnp.array(np.loadtxt('MATLAB/Fscale.txt', delimiter=','))
invV = jnp.array(np.loadtxt('MATLAB/invV.txt', delimiter=','))
rk4a = jnp.array(np.loadtxt('MATLAB/rk4a.txt', delimiter=','))
rk4b = jnp.array(np.loadtxt('MATLAB/rk4b.txt', delimiter=','))
rk4c = jnp.array(np.loadtxt('MATLAB/rk4c.txt', delimiter=','))
rx = jnp.array(np.loadtxt('MATLAB/rx.txt', delimiter=','))
V = jnp.array(np.loadtxt('MATLAB/V.txt', delimiter=','))
vmapM = jnp.array(np.loadtxt('MATLAB/vmapM.txt', delimiter=',', dtype=int) - 1)
vmapP = jnp.array(np.loadtxt('MATLAB/vmapP.txt', delimiter=',', dtype=int) - 1)
vmapI = jnp.array(np.loadtxt('MATLAB/vmapI.txt', delimiter=',', dtype=int) - 1)
vmapO = jnp.array(np.loadtxt('MATLAB/vmapO.txt', delimiter=',', dtype=int) - 1)
mapI = jnp.array(np.loadtxt('MATLAB/mapI.txt', delimiter=',', dtype=int) - 1)
mapO = jnp.array(np.loadtxt('MATLAB/mapO.txt', delimiter=',', dtype=int) - 1)
x = jnp.array(np.loadtxt('MATLAB/x.txt', delimiter=','))
N = int(np.loadtxt('MATLAB/N.txt', delimiter=','))
K = int(np.loadtxt('MATLAB/K.txt', delimiter=','))
nx = jnp.array(np.loadtxt('MATLAB/nx.txt', delimiter=','))

EToE = jnp.array(np.loadtxt('MATLAB/EToE.txt', delimiter=',') - 1).astype(int)

Np = N + 1
Nfp = 1
Nfaces = 2
a = 1
alpha = 1

modes = 4
Basis = np.zeros((modes, Np, K))
# for i in range(1, int(modes/2) + 1):
#     Basis[2*i-2, :] = np.sin(np.pi * 2 * i * x)
#     Basis[2*i-1, :] = np.cos(np.pi * 2 * i * x)

# Basis = jnp.asarray(Basis)

FinalTime = 0.4

# compute time step size
xmin = np.min(np.abs(x[1, :] - x[2, :]))
CFL = 1
gamma = 1.4
c = jnp.sqrt(2*gamma)
dt = CFL / c*xmin
dt = .5*dt

# print(dt)

Nsteps = int(np.ceil(FinalTime / dt))
dt = FinalTime / Nsteps
# dt = np.round(dt*100) / 100

nt_step_train = int(np.ceil(Nsteps/10)) + 1  # including the initial condition
nt_step_test = Nsteps + 1

# dt = 0.01

print(dt, Nsteps)


def minmod(v: jnp.ndarray) -> jnp.ndarray:
  """Applies minmod to v
  """
  m = v.shape[0]
  s = jnp.sum(jnp.sign(v), axis=0) / m
  v_mask = jnp.array(
      [jnp.where(jnp.abs(s) == 1, jnp.abs(v[i]), 0) for i in range(v.shape[0])])
  m_fn = s*jnp.min(v_mask, axis=0)
  return m_fn


def slope_limit_lin(ul, xl, vm1, v0, vp1):
  """Apply slope limiter on linear function ul(Np,1) on x(Np,1)
    (vm1, v0, vp1) are cell averages left, center, right
  """
  # compute geometric measures
  h = xl[Np - 1, :] - xl[0, :]
  x0 = jnp.ones((Np, 1))*(xl[0, :] + h/2)

  hN = jnp.ones((Np, 1))*h

  # limit function
  ux = (2/hN)*(Dr@ul)

  ulimit = jnp.ones((Np, 1))*v0 + (xl-x0)*jnp.ones(
      (Np, 1))*minmod(jnp.array([ux[0, :], (vp1-v0) / h, (v0-vm1) / h]))

  return ulimit


def slope_limit_n(u):
  """Apply slope limiter 
    
    .. math::
        \Pi^N

    to u assuming u is an Nth order polynomial
  """
  # Compute Cell Averages
  uh = invV @ u
  uh = uh.at[1:Np, :].set(0)
  uavg = V @ uh
  v = uavg[0, :]

  # Apply Slope Limiter
  eps0 = 1e-8

  # find end values of each element
  ue1 = u[0, :]
  ue2 = u[-1, :]

  # find cell averages
  vk = v
  vkm1 = jnp.array([v[0], *v[0:K - 1]])
  vkp1 = jnp.array([*v[1:K], v[K + 1]])

  # apply reconstruction
  ve1 = vk - minmod(jnp.array([vk - ue1, vk - vkm1, vkp1 - vk]))
  ve2 = vk + minmod(jnp.array([ue2 - vk, vk - vkm1, vkp1 - vk]))

  # check if elements require limiting
  ids = jnp.logical_or(jnp.abs(ve1 - ue1) > eps0, jnp.abs(ve2 - ue2) > eps0)
  # create piecewise linear solution for limiting
  uhl = invV @ u
  uhl = uhl.at[2:Np, :].set(0)
  ul = V @ uhl

  # apply slope limiter
  ulimit = jnp.where(ids, slope_limit_lin(ul, x, vkm1, vk, vkp1), u)

  return ulimit


def EulerRHS1D(rho, rhou, Ener, bc):
  """Evalueate RHS flux for 1D Euler Equations
  """
  # compute max velocity for LF Flux
  pres = (gamma-1)*(Ener - 0.5*(rhou)**2 / rho)
  cvel = jnp.sqrt(gamma*pres / rho)
  lm = jnp.abs(rhou / rho) + cvel

  # compute flux
  rhof = rhou
  rhouf = rhou**2 / rho + pres
  Enerf = (Ener+pres)*rhou / rho

  # compute jumps
  drho = jnp.reshape((rho.ravel()[vmapM] - rho.ravel()[vmapP]), (Nfp*Nfaces, K))
  drhou = jnp.reshape((rhou.ravel()[vmapM] - rhou.ravel()[vmapP]),
                      (Nfp*Nfaces, K))
  dEner = jnp.reshape((Ener.ravel()[vmapM] - Ener.ravel()[vmapP]),
                      (Nfp*Nfaces, K))
  drhof = jnp.reshape((rhof.ravel()[vmapM] - rhof.ravel()[vmapP]),
                      (Nfp*Nfaces, K))
  drhouf = jnp.reshape((rhouf.ravel()[vmapM] - rhouf.ravel()[vmapP]),
                       (Nfp*Nfaces, K))
  dEnerf = jnp.reshape((Enerf.ravel()[vmapM] - Enerf.ravel()[vmapP]),
                       (Nfp*Nfaces, K))
  LFc = jnp.reshape(
      jnp.max(jnp.array([lm.ravel()[vmapM],
                         lm.ravel()[vmapP]]), 0), (Nfp*Nfaces, K))

  # compute flux at interfaces
  drhof = jnp.reshape(
      (nx.ravel()*drhof.ravel() / 2 - LFc.ravel() / 2*drho.ravel()),
      (Nfp*Nfaces, K))
  drhouf = jnp.reshape(
      (nx.ravel()*drhouf.ravel() / 2 - LFc.ravel() / 2*drhou.ravel()),
      (Nfp*Nfaces, K))
  dEnerf = jnp.reshape(
      (nx.ravel()*dEnerf.ravel() / 2 - LFc.ravel() / 2*dEner.ravel()),
      (Nfp*Nfaces, K))

  # BC's for shock tube
  rhoin = bc[0, 0]
  rhouin = bc[0, 1]
  Enerin = bc[0, -1]
  pin = Enerin*(gamma-1)
  rhoout = bc[-1, 0]
  rhouout = bc[-1, 1]
  Enerout = bc[-1, -1]
  pout = Enerout*(gamma-1)

  # set fluxes at inflow/outflow
  rhofin = rhouin
  rhoufin = rhouin**2 / rhoin + pin
  Enerfin = (pin / (gamma-1) + 0.5*rhouin**2 / rhoin + pin)*rhouin / rhoin
  lmI = lm.ravel()[vmapI] / 2
  nxI = nx.ravel()[mapI]
  drho = jnp.reshape(
      drho.ravel().at[mapI].set(
          jnp.squeeze(nxI*(rhof.ravel()[vmapI] - rhofin) / 2 - lmI*
                      (rho.ravel()[vmapI] - rhoin))), (Nfp*Nfaces, K))
  drhou = jnp.reshape(
      drhou.at[mapI].set(
          jnp.squeeze(nxI*(rhouf.ravel()[vmapI] - rhoufin) / 2 - lmI*
                      (rhou.ravel()[vmapI] - rhouin))), (Nfp*Nfaces, K))
  dEner = jnp.reshape(
      dEner.at[mapI].set(
          jnp.squeeze(nxI*(Enerf.ravel()[vmapI] - Enerfin) / 2 - lmI*
                      (Ener.ravel()[vmapI] - Enerin))), (Nfp*Nfaces, K))

  rhofout = rhouout
  rhoufout = rhouout**2 / rhoout + pout
  Enerfout = (pout /
              (gamma-1) + 0.5*rhouout**2 / rhoout + pout)*rhouout / rhoout
  lmO = lm.ravel()[vmapO] / 2
  nxO = nx.ravel()[mapO]
  drho = jnp.reshape(
      drho.ravel().at[mapO].set(
          jnp.squeeze(nxO*(rhof.ravel()[vmapO] - rhofout) / 2 - lmO*
                      (rho.ravel()[vmapO] - rhoout))), (Nfp*Nfaces, K))
  drhou = jnp.reshape(
      drhou.at[mapO].set(
          jnp.squeeze(nxO*(rhouf.ravel()[vmapO] - rhoufout) / 2 - lmO*
                      (rhou.ravel()[vmapO] - rhouout))), (Nfp*Nfaces, K))
  dEner = jnp.reshape(
      dEner.at[mapO].set(
          jnp.squeeze(nxO*(Enerf.ravel()[vmapO] - Enerfout) / 2 - lmO*
                      (Ener.ravel()[vmapO] - Enerout))), (Nfp*Nfaces, K))

  # compute rhs of the PDEs
  rhsrho = -rx*(Dr@rhof) + LIFT @ (Fscale*drhof)
  rhsrhou = -rx*(Dr@rhouf) + LIFT @ (Fscale*drhouf)
  rhsEner = -rx*(Dr@Enerf) + LIFT @ (Fscale*dEnerf)
  return rhsrho, rhsrhou, rhsEner


def numerical_solver(u, bc):
  rho = u[..., 0]
  rhou = u[..., 1]
  Ener = u[..., 2]

  # SSP RK 1
  rhsrho, rhsrhou, rhsEner = EulerRHS1D(rho, rhou, Ener, bc)
  rho1 = slope_limit_n(rho + dt*rhsrho)
  rhou1 = slope_limit_n(rhou + dt*rhsrhou)
  Ener1 = slope_limit_n(Ener + dt*rhsEner)

  # SSP RK 2
  rhsrho, rhsrhou, rhsEner = EulerRHS1D(rho1, rhou1, Ener1, bc)
  rho2 = slope_limit_n((3*rho + rho1 + dt*rhsrho) / 4)
  rhou2 = slope_limit_n((3*rhou + rhou1 + dt*rhsrhou) / 4)
  Ener2 = slope_limit_n((3*Ener + Ener1 + dt*rhsEner) / 4)

  # SSP RK 3
  rhsrho, rhsrhou, rhsEner = EulerRHS1D(rho1, rhou1, Ener1, bc)
  rho3 = slope_limit_n((rho + 2*rho2 + 2*dt*rhsrho) / 3)
  rhou3 = slope_limit_n((rhou + 2*rhou2 + 2*dt*rhsrhou) / 3)
  Ener3 = slope_limit_n((Ener + 2*Ener2 + 2*dt*rhsEner) / 3)

  u = jnp.array([rho3, rhou3, Ener3])
  return u


def generate_data(u, Nsteps):
  step_save = 0
  U_data = jnp.zeros((Nsteps, K*(N+1)*3))
  U_data = U_data.at[step_save, :].set(u.flatten())
  bc = u[[0, -1], [0, -1], :]

  for tstep in range(1, Nsteps):  # outer time step loop
    u = numerical_solver(u, bc)
    step_save += 1
    U_data = U_data.at[step_save, :].set(u.flatten())

  return U_data


generate_data_batch = vmap(generate_data, in_axes=(0, None))


def get_shock_init(coefs, x):
  coefs = jnp.sort(jnp.abs(coefs))
  p1, rho1, p4, rho4 = coefs
  # keep speed of sound below sqrt(2*gamma) for CFL
  eps = 1e-8
  p1 = jnp.where(p1 >= 2*rho1, 2*rho1 - eps, p1)
  p4 = jnp.where(p4 >= 2*rho4, 2*rho4 - eps, p4)
  u = jnp.zeros((*x.shape, 3))
  u = u.at[:, :, 0].set(jnp.where(x <= 0.5, rho4, rho1))
  u = u.at[:, :, 2].set(jnp.where(x <= 0.5, p4 / (gamma-1), p1 / (gamma-1)))
  # u = [rho, rhou, Ener]
  return u


### Generate train data
print('Generating train data ......................')
coeffs_train = random.normal(train_seed, (num_train, modes))
# u_batch_train = np.einsum('bi, ikl -> bkl', coeffs_train, Basis)
u_batch_train = vmap(
    get_shock_init, in_axes=(0, None))(coeffs_train, jnp.array(x))
# bc_train = jnp.array([u_batch_train[:,0,0,:], u_batch_train[:,-1,-1,:]])
# bc_train = jnp.moveaxis(bc_train, 1, 0)
train_data = generate_data_batch(u_batch_train, nt_step_train)
print(train_data.shape)
print(train_data.max())
Solution_samples_array = pd.DataFrame({'samples': train_data.flatten()})
Solution_samples_array.to_csv(
    'data/shock/Train_noise_' + str(0.00) + '_d_' + str(num_train) + '_Nt_' +
    str(nt_step_train) + '_K_' + str(K) + '_Np_' + str(N) + '.csv',
    index=False)

### Generate test data
print('Generating test data ......................')
coeffs_test = random.normal(test_seed, (num_test, modes))
# u_batch_test = np.einsum('bi, ikl -> bkl', coeffs_test, Basis)
u_batch_test = vmap(
    get_shock_init, in_axes=(0, None))(coeffs_test, jnp.array(x))
# bc_test = jnp.array([u_batch_train[:,0,0,:], u_batch_train[:,-1,-1,:]])
# bc_test = jnp.moveaxis(bc_test, 1, 0)
test_data = generate_data_batch(u_batch_test, nt_step_test)
print(test_data.shape)
print(test_data.max())
Solution_samples_array = pd.DataFrame({'samples': test_data.flatten()})
Solution_samples_array.to_csv(
    'data/shock/Test_d_' + str(num_test) + '_Nt_' + str(nt_step_test) + '_K_' +
    str(K) + '_Np_' + str(N) + '.csv',
    index=False)

### Generate test data
print('Generating test data ......................')
nt_test = 401*5
test_data = generate_data_batch(u_batch_test, nt_test)
print(test_data.shape)
Solution_samples_array = pd.DataFrame({'samples': test_data.flatten()})
Solution_samples_array.to_csv(
    'data/shock/Test_d_' + str(num_test) + '_Nt_' + str(nt_test) + '_K_' +
    str(K) + '_Np_' + str(N) + '.csv',
    index=False)

# print('='*20 + ' TRAIN NOISE DATA () ' + '='*20)
key_data_noise = random.PRNGKey(3)
U_train = np.reshape(train_data, (num_train, nt_step_train, K, N + 1, 3))
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
    'data/shock/Train_noise_' + str(noise_level) + '_d_' + str(num_train) +
    '_Nt_' + str(nt_step_train) + '_K_' + str(K) + '_Np_' + str(N) + '.csv',
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
