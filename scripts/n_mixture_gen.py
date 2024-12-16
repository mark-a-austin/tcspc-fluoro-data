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
from typing import List, Tuple

PHOTON_COUNT_BOUNDS = (5_000_000, 30_000_000)

BASE_PATH = "../data/synthetic-data/mixed-fluoro"
PHOTON_DATA_PATH = f"{BASE_PATH}/histograms"
EMISSION_DATA_PATH = f"{BASE_PATH}/emission"
MANIFEST_PATH = f"{BASE_PATH}/manifest.csv"

TOTAL_SAMPELES = 10_000


def init_manifest_file(manifest_path: str = MANIFEST_PATH):
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            f.write(
                """This file contains the metadata of the synthetic mixture data. The manifest file will have the following columns:
                - num_components: int -> the number of components (fluorophores) in the mixture
                - photon_data_loc: str -> the location of the histogram photon data
                - emission_data_locs: [str] -> the location of the emission data
                - max_intensity: [float] -> the maximum intensity of the components
                - avg_intensity: [float] -> the average intensity of the components
                - concentrations: [float] -> the concentrations of the components
                - lifetime: [float] -> the lifetime of the components
                - photon_count: [int] -> the photon count of the components
                - emission_min: [float] -> the minimum emission of the components
                - emission_max: [float] -> the maximum emission of the components
                - use_bias: bool -> whether the bias was used
                - use_pde: bool -> whether the pde was used\n"""
            )

            f.write(
                "num_components;photon_data_loc;emission_data_locs;max_intensity;avg_intensity;concentrations;lifetime;photon_count;emission_min;emission_max;use_bias;use_pde\n"
            )


def write_manifest_log(
    num_components: int,
    photon_data_loc: str,
    emission_data_locs: List[str],
    max_intensity: List[float],
    avg_intensity: List[float],
    concentrations: List[float],
    lifetime: List[float],
    photon_count: List[int],
    emission_min: List[float],
    emission_max: List[float],
    use_bias: bool,
    use_pde: bool,
    manifest_path: str = MANIFEST_PATH,
):
    with open(manifest_path, "a") as f:
        f.write(
            f"{num_components};{photon_data_loc};{emission_data_locs};{max_intensity};{avg_intensity};{concentrations};{lifetime};{photon_count};{emission_min};{emission_max};{use_bias};{use_pde}\n"
        )


def generate_single_random_fluoro(
    irf: IRF,
    pde: SpectralSensitivity,
    emission_generator: Emission_Generator,
    use_endogenous: bool = False,
    use_bias: bool = False,
    use_pde: bool = False,
    verbose: bool = False,
):
    if use_endogenous:
        emission_generator.generate_random_quadtraic_emission(verbose=verbose)
    else:
        emission_generator.generate_random_emission(verbose=verbose)
    photon_count = np.random.randint(*PHOTON_COUNT_BOUNDS)
    lifetime = np.round(np.random.uniform(0.1, 10), 2)
    time = dt.datetime.now().strftime("%Y-%m-%d|%H-%M-%S-%f")
    fluo = Tissue_Fluorophore(
        name=f"random_fluoro_{time}",
        spectral_sensitivity_range=pde.red_spad_range,
        spectral_sensitivity=pde.red_spad_sensitivity,
        average_lifetime=lifetime,
        intensity_distribution=emission_generator.spline,
        intensity_range=emission_generator.emission_bounds,
        irf_function=irf.lookup,
        irf_mu_lookup=irf.mu_lookup,
    )

    fluo_data = fluo.generate_data(
        photon_count,
        use_bias=use_bias,
        use_spectral_sensitivity=use_pde,
    )
    max_intensity, avg_intensity = get_max_and_average_peak_intensity_per_channel(
        fluo_data, fluo.bias.get_intensity_matrix_indicies()
    )
    res = {
        "lifetime": lifetime,
        "photon_count": photon_count,
        "max_intensity": max_intensity,
        "avg_intensity": avg_intensity,
        "emission_generator": emission_generator,
        "histogram": fluo_data,
    }
    return res


def create_n_mixture(
    n: int,
    use_bias: bool = False,
    use_pde: bool = False,
    verbose: bool = False,
    manifest_loc: str = MANIFEST_PATH,
    histogram_loc: str = PHOTON_DATA_PATH,
    emission_loc: str = EMISSION_DATA_PATH,
):

    irf = IRF(path=IRF_PATH)
    pde = SpectralSensitivity(path=PDE_PATH)
    time = dt.datetime.now().strftime("%Y-%m-%d|%H-%M-%S-%f")

    fluoros = []
    for i in range(n):
        fluoros.append(
            generate_single_random_fluoro(
                irf,
                pde,
                Emission_Generator(),
                use_endogenous=np.random.choice([True, False]),
                use_bias=False,
                use_pde=False,
                verbose=verbose,
            )
        )

    bounds_min = np.min([f["emission_generator"].emission_bounds[0] for f in fluoros])
    bounds_max = np.max([f["emission_generator"].emission_bounds[1] for f in fluoros])
    total_bounds = (bounds_min, bounds_max)

    bias = Bias(Wavebounds(*total_bounds)).get_bias_matrix().astype(int)
    pde_matrix = pde.get_pde_matrix()

    data = np.zeros(fluoros[0]["histogram"].shape)
    for f in fluoros:
        data += f["histogram"]

    if use_pde:
        data = np.round(data * pde_matrix).astype(int)
    if use_bias:
        data += bias

    if verbose:
        plotting_fluoro = Tissue_Fluorophore(
            name="plotting_fluoro",
            spectral_sensitivity_range=pde.red_spad_range,
            spectral_sensitivity=pde.red_spad_sensitivity,
            average_lifetime=0,
            intensity_distribution=fluoros[0]["emission_generator"].spline,
            intensity_range=total_bounds,
            irf_function=irf.lookup,
            irf_mu_lookup=irf.mu_lookup,
        )

        plotting_fluoro.plot_data(data, time_range=(0, 15), block=True)

    # sort fluoros by lifetime
    fluoros = sorted(fluoros, key=lambda x: x["lifetime"], reverse=True)
    # saving histogram data
    data_loc = f"{histogram_loc}/{time}.npz"
    np.savez_compressed(data_loc, data=data)
    # saving emission data
    emissions_locs = []
    for i, f in enumerate(fluoros):
        f["emission_generator"].save_emission_metadata(f"{time}_{i}", emission_loc)
        emissions_locs.append(f"{emission_loc}/{time}_{i}_emission.npz")
    max_intensity = [f["max_intensity"] for f in fluoros]
    avg_intensity = [f["avg_intensity"] for f in fluoros]
    total_peak = np.sum([f["max_intensity"] for f in fluoros])
    concentations = [f["max_intensity"] / total_peak for f in fluoros]
    lifetimes = [f["lifetime"] for f in fluoros]
    photon_counts = [f["photon_count"] for f in fluoros]
    emission_min = [f["emission_generator"].emission_bounds[0] for f in fluoros]
    emission_max = [f["emission_generator"].emission_bounds[1] for f in fluoros]

    write_manifest_log(
        n,
        data_loc,
        emissions_locs,
        max_intensity,
        avg_intensity,
        concentations,
        lifetimes,
        photon_counts,
        emission_min,
        emission_max,
        use_bias,
        use_pde,
        manifest_loc,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("--num_components", type=int, default=2)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--use_pde", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--samples", type=int, default=TOTAL_SAMPELES)
    args = parser.parse_args()

    os.makedirs(PHOTON_DATA_PATH, exist_ok=True)
    os.makedirs(EMISSION_DATA_PATH, exist_ok=True)

    init_manifest_file()

    if args.test:
        create_n_mixture(
            args.num_components,
            use_bias=args.use_bias,
            use_pde=args.use_pde,
            verbose=args.verbose,
            manifest_loc=MANIFEST_PATH,
            histogram_loc=PHOTON_DATA_PATH,
            emission_loc=EMISSION_DATA_PATH,
        )
        sys.exit(0)

    for i in tqdm(range(args.samples)):
        try:
            create_n_mixture(
                args.num_components,
                use_bias=args.use_bias,
                use_pde=args.use_pde,
                verbose=args.verbose,
                manifest_loc=MANIFEST_PATH,
                histogram_loc=PHOTON_DATA_PATH,
                emission_loc=EMISSION_DATA_PATH,
            )
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
