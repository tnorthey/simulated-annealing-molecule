
    def simulated_annealing_zmat(
        self,
        starting_xyz,
        target_pcd,
        qvector,
        starting_temp=0.2,
        nsteps=10000,
        step_size=0.1,
    ):
        """simulated annealing minimisation to target_pcd_array"""
        ##=#=#=# DEFINITIONS #=#=#=##
        ## start.xyz, reference.xyz ##
        _, _, atomlist, reference_xyz = m.read_xyz("xyz/reference.xyz")
        atomic_numbers = [m.periodic_table(symbol) for symbol in atomlist]
        reference_iam = x.iam_calc(atomic_numbers, reference_xyz, qvector)
        # chemcoord molecule dataframe
        df = pd.DataFrame(
            {
                "atom": atomlist,
                "x": starting_xyz[:, 0],
                "y": starting_xyz[:, 1],
                "z": starting_xyz[:, 2],
            }
        )
        molecule = chemcoord.Cartesian(df)
        starting_zmat = molecule.get_zmat()
        # other definitions...
        natoms = starting_xyz.shape[0]  # number of atoms
        ## q-vector, atomic, and pre-molecular IAM contributions ##
        qlen = len(qvector)  # length of q-vector
        aa, bb, cc = self.read_iam_coeffs()
        atomic_total, pre_molecular = self.atomic_pre_molecular(
            atomic_numbers, qvector, aa, bb, cc
        )
        qpi = qvector / np.pi  # used with np.sinc function inside loop
        target_pcd /= np.max(np.abs(target_pcd))  # normalise target to abs_max
        ##=#=#=# END DEFINITIONS #=#=#=#

        ##=#=#=# INITIATE LOOP VARIABLES #=#=#=#=#
        xyz = starting_xyz
        zmat = starting_zmat
        i, c = 0, 0
        chi2, chi2_best = 1e9, 1e10
        ##=#=#=# END INITIATE LOOP VARIABLES #=#=#
        while i < nsteps:
            i += 1  # count steps

            ##=#=#=#=# TEMPERATURE #=#=#=#=#=#=#=#=##
            tmp = 1 - i / nsteps  # this is prop. to how far the molecule moves
