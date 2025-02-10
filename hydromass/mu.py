import numpy as np

def mean_molecular_weights(infile, abund='aspl', Zs=0.3):
    '''
    Calculation of mean molecular weights given an abundance table and a metallicity. The function reads the abundance table from a file located in the hydromass databse and that is copied from the XSPEC abundance table implementation. The abundance table is then used to calculate the mean molecular weight, the number ratio of electrons to ions, and the mean molecular weight per electron.

    :param infile: Path to the XSPEC file containing the abundance tables
    :type infile: str
    :param abund: Name of abundance table to be used. Available abundance tables are 'feld', 'angr', 'aneb', 'grsa', 'wilm', 'lodd', 'aspl', 'lpgp', 'lpgs', and 'felc'. Defaults to 'aspl'.
    :type abund: str
    :param Zs: Metallicity of the plasma with respect to Solar. Defaults to 0.3.
    :type Zs: float
    :return: Mean molecular weight, ratio of number densities of electrons to ions, mean molecular weight per electron
    :type float, float, float
    '''

    fin = open(infile, 'r')
    lin = fin.readlines()
    fin.close()

    atomic_weights = np.array([1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.086, 30.974,
                      32.066, 35.453, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38])

    atomic_numbers = np.arange(1, 31)

    nele = len(atomic_weights)

    Ns = np.empty(nele)

    for nl, line in enumerate(lin):

        if abund in line and nl<12:

            tl = line.split()

            for i in range(nele):

                Ns[i] = float(tl[i+1])

    abvec = np.ones(nele) * Zs # Abundance corrected for ratio to Solar metallicity

    abvec[0] = 1 # Abundance of Hydrogen and Helium is set to 1

    abvec[1] = 1

    tot = np.sum(Ns * atomic_weights * abvec)

    Xj = Ns * atomic_weights / tot

    X = Xj[0]

    Y = Xj[1]

    Z = np.sum(Xj[2:])

    print('X:', X)
    print('Y:', Y)
    print('Z:', Z)

    mup = 1 / np.sum((1. + atomic_numbers) / atomic_weights * abvec * Xj) # Eq. 10.20 of Carroll

    print('Mean molecular weight:', mup)

    nhc = np.sum(atomic_numbers * Ns * abvec) / Ns[0] # electrons / Hydrogen nuclei

    print('Number ratio of electrons to H:', nhc)

    mu_e = 2 / (1 + X) # mean molecular weight per electron

    print('Mean molecular weight per electron:', mu_e)

    return mup, nhc, mu_e
