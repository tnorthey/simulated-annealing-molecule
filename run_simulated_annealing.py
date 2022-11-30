import numpy as np
import molecule
import sys
from timeit import default_timer
from numpy.random import random_sample as random

start = default_timer()

# command line arguments
qmax = float(sys.argv[1])
qlen = int(sys.argv[2])
step_size = float(sys.argv[3])
nsteps = int(sys.argv[4])
tstart = int(sys.argv[5])
ntsteps = int(sys.argv[6])
starting_temp = float(sys.argv[7])
start_xyz_file = str(sys.argv[8])

m = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sa = molecule.Simulated_Annealing()

# definitions
qmin = 0.1
qvector = np.linspace(qmin, qmax, qlen, endpoint=True)

non_h_indices = [0, 1, 2, 3, 4, 5]

_, _, atomlist, starting_xyz = m.read_xyz(start_xyz_file)
_, _, atomlist, reference_xyz = m.read_xyz("xyz/reference.xyz")
atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
compton_array = x.compton_spline(atomic_numbers, qvector)
reference_iam = x.iam_calc_compton(atomic_numbers, reference_xyz, qvector, compton_array)
nmfile = "nm/chd_normalmodes.txt"
natoms = reference_xyz.shape[0]
displacements = nm.read_nm_displacements(nmfile, natoms)

# "experiment" target percent diff
_, _, _, target_xyz_array = m.read_xyz_traj("xyz/chd_target_traj.xyz", ntsteps)

# define zero arrays 
final_xyz_array = np.zeros((natoms, 3, ntsteps))
final_chi2_array = np.zeros(ntsteps)
final_pcd_array = np.zeros((qlen, ntsteps))
target_pcd_array = np.zeros((qlen, ntsteps))
thresh = 1e-6
max_restarts = 10

for t in range(tstart, ntsteps):
    print('time-step: %i' % t)
    target_iam = x.iam_calc_compton(atomic_numbers, target_xyz_array[:, :, t], qvector, compton_array)
    target_pcd_array[:, t] = 100 * (target_iam / reference_iam - 1)
    chi2_best = 1e9  # arbitarily large starting value

    target_pcd = target_pcd_array[:, t]
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

    for i in range(3):
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
    _, r = sa.rmsd_kabsch(xyz_best, target_xyz_array[:, :, t], non_h_indices)
    final_xyz_array[:, :, t] = np.dot(xyz_best, r.as_matrix())
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
