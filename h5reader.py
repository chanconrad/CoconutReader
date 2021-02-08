"""
David's Reader for VERTEX HDF5 hydro output files

Usage:
Run in directory
y = h5reader.hydrof()
y.den(index = 3) gives 3rd time step of density
"""

from glob import glob
import h5py
import numpy as np
import os


# For Prometheus, we need to throw out alpha, beta1, beta2, beta3, the Prometheus files don't have them


class hydrof:
    def __init__(self, model, directory=None, index=0):
        """
        alpha                lapse function
        beta1                shift vector (radial)
        beta2                shift vector (meridional)
        beta3                shift vector (zonal)
        bx                    radius of zone boundaries
        by                    inclination (theta) of zone boundaries
        bz                    azimuth (phi) of zone boundaries
        cpo                    chemical potentials
        den                    density
        dnu                    neutrino number density (lab)
        dt                    hydro timestep interval
        ene                    specific energy
        enu                    neutrino energy density
        eph                    Lapse function (grav. potential)
        fnu                    neutrino energy flux (lab)
        gac                    adiabatic index
        gam                    Gamma factor (grav. potential)
        gpo                    grav. pot. in hydro grid
        ish                    shock zones
        nstep                hydro timestep number
        phi                    conformal factor
        pnu                    neutrino pressure
        pre                    pressure
        qen                    Quell-ene
        qmo                    Quell-mom
        qye                    Quell-Ye
        restmass_version    Energy normalization: different version for the
                            subtraction of rest masses from the energy used in
                            PPM:
                            0: uses energy defined as in EoS
                            1: subtracts from EoS energy the baryon rest masses,
                            assuming that heavy elements have the mass of
                            fe56
                            2: subtracts from EoS energy the baryon rest
                            masses
                            3: subtracts from EoS energy the baryon and
                            unpaired electron rest masses. Caution! This
                            version violates energy!
        sto                    entropy per baryon
        tem                    temperature
        tgm                    Enclosed gravitational mass
        time                physical time
        tm                    Enclosed baryonic mass
        vex                    velocity in radial direction
        vey                    velocity in theta direction
        vez                    velocity in phi direction
        xcart                X coordinate (Cartesian)
        xnu                    composition (mass fractions)
        xzl                    radius at left rim in hydro grid
        xzn                    radius of zone center
        xzr                    radius at right rim in hydro grid
        ycart                Y coordinate (Cartesian)
        yzl                    inclination (theta) at left rim in hydro grid
        yzn                    inclination (theta) of zone center
        yzr                    inclination (theta) at right rim in hydro grid
        zcart                Z coordinate (Cartesian)
        zzl                    azimuth (phi) at left rim in hydro grid
        zzn                    azimuth (phi) of zone center
        zzr                    azimuth (phi) at right rim in hydro grid
        """

        # A list of all the variable names.
        self.variables = [
            "alpha",
            "beta1",
            "beta2",
            "beta3",
            "bx",
            "by",
            "bz",
            "cpo",
            "den",
            "dnu",
            "dt",
            "ene",
            "enu",
            "eph",
            "fnu",
            "gac",
            "gam",
            "gpo",
            "ish",
            "nstep",
            "phi",
            "pnu",
            "pre",
            "qen",
            "qmo",
            "qye",
            "restmass_version",
            "sto",
            "tem",
            "tgm",
            "time",
            "tm",
            "vex",
            "vey",
            "vez",
            "xcart",
            "xnu",
            "xzl",
            "xzn",
            "xzr",
            "ycart",
            "yzl",
            "yzn",
            "yzr",
            "zcart",
            "zzl",
            "zzn",
            "zzr",
        ]

        if directory is None:
            directory = "."

        search = os.path.join(directory, "{}.o*".format(model))
        print(f"Searching {search}")
        hdf5 = glob(search)

        if len(hdf5) == 0:
            raise FileNotFoundError(
                f"No dump files found in the directory ({directory})"
            )

        hdf5.sort()
        steps = []
        for name in hdf5:
            try:
                steps += [h5py.File(name, "r")]
            except FileNotFoundError:
                print("Warning: Could not open", name)
        self.steps = steps

        # Creats a mapping key between indices of sub-timesteps and major-steps.
        key = {}
        M = -1
        N = -1
        for i in range(len(self.steps)):
            length = len(self.steps[i])
            N = N + 1
            for j in range(length):
                M = M + 1
                key[M] = N
        self.key = key

        # Create a list of the group (sub-timestep) names and print them.
        self.groups = []
        for i in range(len(self.steps)):
            for j in self.steps[i]:
                # print(j)
                self.groups.append(j)
        self.index = index
        self.ngroups = len(self.groups)
        print("Number of timesteps: ", self.ngroups)

    # Method for changing the current index value
    def set_index(self, index, silent=True):
        self.index = index
        if not silent:
            print("Current index is now: ", self.index)

    # ------------------------------------------------------------------------------
    # Here we define methods to load a single variable at a given timestep.
    # The timestep loaded by default is the current index. But any timestep can be
    # loaded by providing the keyword argument: index=....

    def read_array(self, var, **kwargs):
        ok = True
        if bool(kwargs) is not False:
            index = kwargs["index"]
            if isinstance(index, int) and 0 <= index <= self.ngroups - 1:
                self.index = index
            else:
                print(
                    "Error: Please choose an integer index between 0 and ",
                    self.ngroups - 1,
                )
                ok = False
        if ok:
            try:
                tmp = self.steps[self.key[self.index]][
                    self.groups[self.index] + f"/{var}"
                ]
                value = np.empty(tmp.shape)
                tmp.read_direct(value)
                value = np.transpose(value)
                return value
            except KeyError:
                print(f"Error: {var} not found")

    def __getattr__(self, var, **kwargs):
        return self.read_array(var, **kwargs)

    # ------------------------------------------------------------------------------
    # This method loads all the variables in all the timesteps ---------------------

    def everything(self):
        print("Loading all timesteps...")
        # timestep = {}
        timestep = []
        T = -1
        for ii in range(len(self.steps)):  # for all coarse steps (by index)
            for i in self.steps[ii]:  # for all substeps in the course step
                T = T + 1
                D = {}
                for j in self.steps[ii][i]:  # for all variables names in the substep
                    tmp = self.steps[ii][i + "/" + j]  # set the hdf5 path as a variable
                    value = np.empty(
                        tmp.shape
                    )  # create an empty array with the size/shape required
                    tmp.read_direct(
                        value
                    )  # read the values from the hdf5 path into the empty array
                    value = np.transpose(value)  # reverse index ordering
                    D[j] = value  # create sub-dictionary of the variables + values
                # timestep[T]=D                        # add into a parent dictionary, containting each sub-step
                timestep += [D]

        self.timestep = timestep
        print("Done.")
