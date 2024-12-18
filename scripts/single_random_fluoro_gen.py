import sys
import numpy as np
import seaborn as sns
import datetime as dt
import time
import os

sys.path.append("../src/")
from Intensity_PDF import Wavebounds
from Tissue_Fluorophore import Tissue_Fluorophore
from irf_function import IRF
from spectral_sensitivity import SpectralSensitivity
from random_emission_generator import Emission_Generator
from bias import Bias
from typing import Tuple
from visualisation_utils import (
    save_data,
    save_peak_intensities,
    save_bias_data,
    plot_peak_intensity_per_channel,
    data_and_irf_inspection,
    single_data_and_irf_inspection,
    get_max_and_average_peak_intensity_per_channel,
    get_peak_intensity_per_channel,
)
from scipy import signal
from scipy import interpolate as interp
from path_vars import IRF_PATH, PDE_PATH
from argparse import ArgumentParser
from tqdm import tqdm


PHOTON_COUNT_BOUNDS = (5_000_000, 30_000_000)

BASE_PATH = "../data/synthetic-data/single-fluoro-endogenous"
PHOTON_DATA_PATH = f"{BASE_PATH}/histograms"
EMISSION_DATA_PATH = f"{BASE_PATH}/emission"
MANIFEST_PATH = f"{BASE_PATH}/manifest.csv"

TOTAL_SAMPELES = 10_000


def add_entry_to_manifest(
    name: str,
    photon_count: int,
    lifetime: float,
    emission_bounds: Tuple[int, int],
    max_intensity: float,
    avg_intensity: float,
    path: str = MANIFEST_PATH,
    photon_data_loc: str = PHOTON_DATA_PATH,
    emission_data_loc: str = EMISSION_DATA_PATH,
    contains_bias: bool = False,
    uses_pde: bool = False,
) -> None:
    """
    This function will create a manifest file that will store the metadata of the generated data
    The manifest file will have the following columns:
    - photon_data_loc : str -> the location of the histogram photon data
    - emission_data_loc : str -> the location of the emission data
    - max_intensity : float -> the maximum intensity of the fluorophore (a.u.)
    - avg_intensity : float -> the average intensity of the fluorophore (a.u.)
    - photon_count : int -> the number of photons generated
    - lifetime : float -> the lifetime of the fluorophore (ns)
    - emission_min : int -> the minimum wavelength of the emission distribution (nm)
    - emission_max : int -> the maximum wavelength of the emission distribution (nm)
    """

    if not os.path.exists(path):
        with open(path, "a") as f:
            # Write the info of each column like the comments
            f.write(
                """This file contains the metadata of the generated single fluorophore data. The manifest file will have the following columns:
                - photon_data_loc : str -> the location of the histogram photon data
                - emission_data_loc : str -> the location of the emission data
                - max_intensity : float -> the maximum intensity of the fluorophore (a.u.)
                - avg_intensity : float -> the average intensity of the fluorophore (a.u.)
                - photon_count : int -> the number of photons generated
                - lifetime : float -> the lifetime of the fluorophore (ns)
                - emission_min : int -> the minimum wavelength of the emission distribution (nm)
                - emission_max : int -> the maximum wavelength of the emission distribution (nm)
                - contains_bias : bool -> if the data has bias added to it
                - uses_pde : bool -> if pde matrix was applied to the data\n"""
            )
            # Write the header
            f.write(
                "photon_data_loc,emission_data_loc,max_intensity,avg_intensity,photon_count,lifetime,emission_min,emission_max,contains_bias,uses_pde\n"
            )

    with open(path, "a") as f:
        data_loc = f"{photon_data_loc}/{name}.npz"
        emission_loc = f"{emission_data_loc}/{name}_emission.npz"
        f.write(
            f"{data_loc},{emission_loc},{max_intensity},{avg_intensity},{photon_count},{lifetime},{emission_bounds[0]},{emission_bounds[1]},{contains_bias},{uses_pde}\n"
        )


def generate_random_fluoro(
    irf: IRF,
    pde: SpectralSensitivity,
    emission_gen: Emission_Generator,
    photon_data_path: str,
    emission_data_path: str,
    manifest_path: str,
    without_bias: bool = False,
    without_pde: bool = False,
    use_endogenous: bool = True,
    verbose: bool = False,
):
    if not use_endogenous:
        emission_gen.generate_random_emission(verbose=verbose)
    else:
        emission_gen.generate_random_quadtraic_emission(verbose=verbose)
    photon_count = np.random.randint(*PHOTON_COUNT_BOUNDS)
    lifetime = np.round(np.random.uniform(0.1, 10), 2)
    time = dt.datetime.now().strftime("%Y-%m-%d|%H-%M-%S-%f")

    fluo = Tissue_Fluorophore(
        name=time,
        spectral_sensitivity_range=pde.red_spad_range,
        spectral_sensitivity=pde.red_spad_sensitivity,
        average_lifetime=lifetime,
        intensity_distribution=emission_gen.spline,
        intensity_range=emission_gen.emission_bounds,
        irf_function=irf.lookup,
        irf_mu_lookup=irf.mu_lookup,
    )
    fluo_data = fluo.generate_data(
        photon_count,
        use_bias=not without_bias,
        use_spectral_sensitivity=not without_pde,
    )
    max_intensity, avg_intensity = get_max_and_average_peak_intensity_per_channel(
        fluo_data, fluo.bias.get_intensity_matrix_indicies()
    )
    fluo.save_data(f"{photon_data_path}/{time}", photon_count, irf, with_metadata=False)
    emission_gen.save_emission_metadata(time, emission_data_path)
    add_entry_to_manifest(
        time,
        photon_count,
        lifetime,
        emission_gen.emission_bounds,
        max_intensity,
        avg_intensity,
        manifest_path,
        photon_data_loc=photon_data_path,
        emission_data_loc=emission_data_path,
        contains_bias=not without_bias,
        uses_pde=not without_pde,
    )

    if verbose:
        print(f"Photon count: {photon_count:_}")
        print(f"Lifetime: {lifetime}")
        print(f"Time: {time}")
        print(f"Emission bounds: {emission_gen.emission_bounds}")
        print(f"Max intensity: {max_intensity}")
        print(f"Avg intensity: {avg_intensity}")
        fluo.plot_data(block=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--verbose",
        help="This flag is used to enable verbose mode",
        action="store_true",
    )
    parser.add_argument(
        "--test", help="This flag is used to enable test mode", action="store_true"
    )
    parser.add_argument(
        "--samples",
        help="The number of samples to generate",
        type=int,
        default=TOTAL_SAMPELES,
    )
    parser.add_argument(
        "--without_bias",
        help="This flag is used to disable the bias added to the data",
        action="store_true",
    )
    parser.add_argument(
        "--without_pde",
        help="This flag is used to disable the PDE",
        action="store_true",
    )

    parser.add_argument(
        "--random_artefacts",
        help="This flag is enable the data generation will randomly turn on and off both the bias and the PDE",
        action="store_true",
    )

    parser.add_argument(
        "--use_endogenous",
        help="This flag is used to enable the use of endogenous fluorophore emission",
        action="store_true",
    )

    args = parser.parse_args()
    verbose = args.verbose
    test = args.test

    irf = IRF(path=IRF_PATH)
    pde = SpectralSensitivity(path=PDE_PATH)
    emission_gen = Emission_Generator()

    samples = args.samples if args.samples else TOTAL_SAMPELES

    # Ensure the directories exist
    os.makedirs(BASE_PATH, exist_ok=True)
    os.makedirs(PHOTON_DATA_PATH, exist_ok=True)
    os.makedirs(EMISSION_DATA_PATH, exist_ok=True)

    if args.random_artefacts:
        bias_choices = np.random.choice([True, False], size=samples)
        pde_choices = np.random.choice([True, False], size=samples)
    else:
        bias_choices = np.full(samples, args.without_bias)
        pde_choices = np.full(samples, args.without_pde)

    # Generate 10_000 single fluorophore data sets
    if not test:
        for i in tqdm(range(samples)):
            try:
                generate_random_fluoro(
                    irf,
                    pde,
                    emission_gen,
                    PHOTON_DATA_PATH,
                    EMISSION_DATA_PATH,
                    MANIFEST_PATH,
                    bias_choices[i],
                    pde_choices[i],
                    args.use_endogenous,
                    verbose,
                )
            except Exception as e:
                continue
            # finally:
            #     time.sleep(0.1)
    else:

        generate_random_fluoro(
            irf,
            pde,
            emission_gen,
            PHOTON_DATA_PATH,
            EMISSION_DATA_PATH,
            MANIFEST_PATH,
            bias_choices[0],
            pde_choices[0],
            args.use_endogenous,
            verbose,
        )
