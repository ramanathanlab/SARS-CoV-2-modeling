import time
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import MDAnalysis
from MDAnalysis.analysis import distances, rms, align

PathLike = Union[str, Path]


def write_contact_map(
    h5_file: h5py.File,
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    vals: Optional[List[np.ndarray]] = None,
):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # list of np arrays of shape (2 * X) where X varies
    data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
    h5_file.create_dataset(
        "contact_map",
        data=data,
        dtype=h5py.vlen_dtype(np.dtype("int16")),
        fletcher32=True,
        chunks=(1,) + data.shape[1:],
    )

    # Write optional values field for contact map. Could contain CA-distances.
    if vals is not None:
        data = ragged(vals)
        h5_file.create_dataset(
            "contact_map_values",
            data=data,
            dtype=h5py.vlen_dtype(np.dtype("float32")),
            fletcher32=True,
            chunks=(1,) + data.shape[1:],
        )


def write_point_cloud(h5_file: h5py.File, point_cloud: np.ndarray):
    h5_file.create_dataset(
        "point_cloud",
        data=point_cloud,
        dtype="float32",
        fletcher32=True,
        chunks=(1,) + point_cloud.shape[1:],
    )


def write_rmsd(h5_file: h5py.File, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def write_h5(
    save_file: PathLike,
    rmsds: List[float],
    rows: List[np.ndarray],
    cols: List[np.ndarray],
    positions: np.ndarray,
):
    """Saves data to h5 file.

    Parameters
    ----------
    save_file : PathLike
        Path of output h5 file used to save datasets.
    rmsds : List[float]
        Stores rmsd data.
    rows : List[np.ndarray]
        rows[i] represents the row indices of a contact map where a 1 exists.
    cols : List[np.ndarray]
        cols[i] represents the column indices of a contact map where a 1 exists.
    positions : np.ndarray
        XYZ coordinate data in the shape: (N, 3, num_residues).
    """
    with h5py.File(save_file, "w", swmr=False) as h5_file:
        write_contact_map(h5_file, rows, cols)
        write_rmsd(h5_file, rmsds)
        write_point_cloud(h5_file, positions)


def traj_to_dset(
    topology: PathLike,
    ref_topology: PathLike,
    traj_file: PathLike,
    save_file: Optional[PathLike] = None,
    cutoff: float = 8.0,
    selection: str = "protein and name CA",
    skip_every: int = 1,
    verbose: bool = False,
    print_every: int = 10,
):
    """Implementation for generating machine learning datasets
    from raw molecular dynamics trajectory data. This function
    uses MDAnalysis to load the trajectory file and given a
    custom atom selection computes contact matrices, RMSD to
    reference state, and the positions (xyz coordinates) of each
    frame in the trajectory.

    Parameters
    ----------
    topology : PathLike
        Path to topology file: CHARMM/XPLOR PSF topology file,
        PDB file or Gromacs GRO file.
    ref_topology : PathLike
        Path to reference topology file for aligning trajectory:
        CHARMM/XPLOR PSF topology file, PDB file or Gromacs GRO file.
    traj_file : PathLike
        Trajectory file (in CHARMM/NAMD/LAMMPS DCD, Gromacs XTC/TRR,
        or generic. Stores coordinate information for the trajectory.
    save_file : Optional[PathLike], default=None
        Path to output h5 dataset file name.
    cutoff : float, default=8.0
        Angstrom cutoff distance to compute contact maps.
    selection : str, default="protein and name CA"
        Selection set of atoms in the protein.
    skip_every : int, default=1
        Only colect data every `skip_every` frames.
    verbose: bool, default=False
        If true, prints verbose output.
    print_every: int, default=10
        Prints update every `print_every` frame.

    Returns
    -------
    Tuple[List] : rmsds, rows, cols
        Lists containing data to be written to HDF5.
    """

    # start timer
    start_time = time.time()

    # Load simulation and reference structures
    sim = MDAnalysis.Universe(str(topology), str(traj_file))
    ref = MDAnalysis.Universe(str(ref_topology))

    if verbose:
        print("Traj length: ", len(sim.trajectory))

    # Align trajectory to compute accurate RMSD and point cloud
    align.AlignTraj(sim, ref, select=selection, in_memory=True).run()

    if verbose:
        print(f"Finish aligning after: {time.time() - start_time} seconds")

    # Atom selection for reference
    atoms = sim.select_atoms(selection)
    # Get atomic coordinates of reference atoms
    ref_positions = ref.select_atoms(selection).positions.copy()
    # Get box dimensions
    box = sim.atoms.dimensions

    rmsds, rows, cols, point_clouds = [], [], [], []

    for i, _ in enumerate(sim.trajectory[::skip_every]):

        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions

        # Compute contact map of current frame (scipy lil_matrix form)
        cm = distances.contact_matrix(positions, cutoff, box=box, returntype="sparse")
        coo = cm.tocoo()
        rows.append(coo.row.astype("int16"))
        cols.append(coo.col.astype("int16"))

        # Compute and store RMSD to reference state
        rmsds.append(
            rms.rmsd(positions, ref_positions, center=True, superposition=True)
        )

        # Store reference atoms point cloud of current frame
        point_clouds.append(positions.copy())

        if verbose:
            if i % print_every == 0:
                msg = f"Frame {i}/{len(sim.trajectory)}"
                msg += f"\trmsd: {rmsds[i]}"
                msg += f"\tshape: {positions.shape}"
                msg += f"\trow shape: {rows[-1].shape}"
                print(msg)

    point_clouds = np.transpose(point_clouds, [0, 2, 1])

    if save_file:
        write_h5(save_file, rmsds, rows, cols, point_clouds)

    if verbose:
        print(f"Duration {time.time() - start_time}s")

    return rmsds, rows, cols, point_clouds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--topology",
        help="Path to topology file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--ref_topology",
        help="Path to reference topology file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--traj_file",
        help="Path to molecular dynamics trajectory file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--save_file",
        help="Path to save output HDF5 file to.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        help="Contact cutoff distances (Angstroms).",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "-s",
        "--selection",
        help="MDAnalysis selection string to use.",
        type=str,
        default="protein and name CA",
    )
    parser.add_argument(
        "--skip_every",
        help="How often to preprocess a frame.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-v", "--verbose", help="Print status to stdout.", action="store_true"
    )
    parser.add_argument(
        "--print_every",
        help="Number of frames between status update.",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    traj_to_dset(**args.__dict__)
