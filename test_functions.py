import numpy as np
import time as t

# my own modules
import molecule

# create class objects
m  = molecule.Molecule()
nm = molecule.Normal_modes()
x = molecule.Xray()
sa = molecule.Simulated_Annealing()

def test_simulated_annealing():
    _, _, atomlist, starting_xyz = m.read_xyz("xyz/start.xyz")
    atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
    nmfile = "nm/chd_normalmodes.txt"
    natoms = 14
    displacements = nm.read_nm_displacements(nmfile, natoms)
    qlen = 99
    qvector = np.linspace(0.1, 12, qlen, endpoint=True)
    starting_iam = x.iam_calc(atomic_numbers, starting_xyz, qvector)
    # "experiment" target percent diff
    tlen = 18
    target_pcd_array = np.zeros((qlen, tlen))
    _, _, _, target_xyz_array = m.read_xyz_traj("xyz/chd_target_traj.xyz", tlen)
    for t in range(tlen):
        target_iam = x.iam_calc(atomic_numbers, target_xyz_array[:, : , t], qvector)
        target_pcd_array[:, t] = 100 * (target_iam / starting_iam - 1)
        target_pcd_array[:, t] /= np.max(np.abs(target_pcd_array[:, t]))  # normalise abs. max value to 1
    target_pcd = target_pcd_array[:, 0]

    starting_temp = 0.2
    nsteps = 10000
    step_size = 0.03

    chi2_best, pcd_best, xyz_best = sa.simulated_annealing_xyz(
        starting_xyz,
        target_pcd,
        qvector,
        starting_temp,
        nsteps,
        step_size,
    )
    print(chi2_best)
    print(pcd_best)
    print(xyz_best)

    m.write_xyz('out.xyz', 'test_functions', atomlist, xyz_best)
start = t.time()
test_simulated_annealing()
end = t.time()
total = float(end - start)
print('time taken: %f' % total)

