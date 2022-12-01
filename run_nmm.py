import numpy as np
import sys
from timeit import default_timer
from numpy.random import random_sample as random
import scipy.io
# my modules
import molecule

start = default_timer()

# initialise class structures
m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sa = molecule.Simulated_Annealing()

# command line arguments
step_size =      float(sys.argv[1])
nsteps =         int(  sys.argv[2])
starting_temp =  float(sys.argv[3])
start_xyz_file = str(  sys.argv[4])

### if you want, load read data... ###
nmm_data = True
if nmm_data:
    # load experiment pcd
    datafile = "data/NMM_exp_dataset.mat"
    mat = scipy.io.loadmat(datafile)
    t_exp = mat["t"]
    q_exp = np.squeeze(mat["q"])
    pcd_exp = mat["iso"]
    errors_exp = mat["iso_stdx"]
    # optionally "clean up" data
    pre_t0_bool, combine_bool = True, True
    if pre_t0_bool:
        i_pre_t0 = 13
        t_exp = t_exp[i_pre_t0:]  # remove before t = 0
        pcd_exp = pcd_exp[:, i_pre_t0:]
    if combine_bool:
        # also combine every 2nd lineout
        t_exp = t_exp[0::2]
        pcd_exp = 0.5 * (pcd_exp[:, 0::2] + pcd_exp[:, 1::2])
    nt = len(t_exp)
    print('Loaded Experiment data with size (Nq, Nt):')
    print(pcd_exp.shape)
    print('q vector:')
    print(q_exp)
    print('t vector:')
    print(t_exp)
###

qvector = q_exp
qlen = len(qvector)

_, _, atomlist, starting_xyz = m.read_xyz(start_xyz_file)
_, _, atomlist, reference_xyz = m.read_xyz("xyz/reference.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
compton_array = x.compton_spline(atomic_numbers, qvector)
reference_iam = x.iam_calc_compton(atomic_numbers, reference_xyz, qvector, compton_array)
nmfile = "nm/nmm_normalmodes.txt"
natoms = reference_xyz.shape[0]
displacements = nm.read_nm_displacements(nmfile, natoms)
print('natoms: %i' % natoms)

# define zero arrays 
final_xyz_array = np.zeros((natoms, 3, nt))
final_chi2_array = np.zeros(nt)
final_pcd_array = np.zeros((qlen, nt))
target_pcd_array = np.zeros((qlen, nt))
thresh = 1e-6
max_restarts = 1

for t in range(nt):
    print('time-step: %i' % t)
    chi2_best = 1e9  # arbitarily large starting value

    target_pcd = pcd_exp[:, t]
    c = 0
    while chi2_best > thresh and c < max_restarts:
        c += 1
        #step_size_ = step_size * (random() + 0.01)
        #print('step_size: %9.8f' % step_size_)
        final_chi2, final_pcd, final_xyz = sa.simulated_annealing_modes(
            starting_xyz,
            displacements,
            target_pcd,
            qvector,
            starting_temp,
            nsteps,
            step_size,
        )
        print("restart: %i, chi2: %9.8f" % (c, final_chi2))
        if final_chi2 < chi2_best:
            chi2_best = final_chi2
            xyz_best = final_xyz
            pcd_best = final_pcd

    starting_xyz = xyz_best

    for i in range(1):
        final_chi2, final_pcd, final_xyz = sa.simulated_annealing_modes(
            starting_xyz,
            displacements,
            target_pcd,
            qvector,
            0.0,
            nsteps,
            step_size / 2,
        )
        print("greedy restart: %i, chi2: %9.8f" % (i, final_chi2))
        if final_chi2 < chi2_best:
            chi2_best = final_chi2
            xyz_best = final_xyz
            pcd_best = final_pcd
        starting_xyz = xyz_best

    # Kabsch rotation to target
    final_xyz_array[:, :, t] = xyz_best
    final_chi2_array[t] = chi2_best
    final_pcd_array[:, t] = pcd_best
    m.write_xyz('out_%s.xyz' % str(t).zfill(2), 't = %i' % t, atomlist, final_xyz_array[:, :, t])


# save to file
data_file = "data.npz"
np.savez(
    data_file,
    step_size=step_size,
    starting_temp=starting_temp,
    nsteps=nsteps,
    qvector=qvector,
    target_pcd_array=target_pcd_array,
    final_pcd_array = final_pcd_array,
    final_xyz_array = final_xyz_array,
    final_chi2_array = final_chi2_array,
)

print("Total time: %3.2f s" % float(default_timer() - start))
