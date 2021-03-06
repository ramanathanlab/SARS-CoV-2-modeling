import h5py
import glob
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[Path, str]


def concatenate_h5(
    input_file_names: List[str],
    output_name: str,
    fields: Optional[List[str]] = None,
):

    if not fields:
        # Peak into first file and collect all the field names
        with h5py.File(input_file_names[0], "r") as h5_file:
            fields = list(h5_file.keys())

    # Should not concatenate this field. Only need 1 copy.
    if "amino_acids" in fields:
        fields.remove("amino_acids")
        add_amino_acids = True
    else:
        add_amino_acids = False

    # Initialize data buffers
    data = {x: [] for x in fields}

    for in_file in input_file_names:
        with h5py.File(in_file, "r", libver="latest") as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    for field in data:
        data[field] = np.concatenate(data[field])

    # Add a single amino acid array
    if add_amino_acids:
        with h5py.File(input_file_names[0], "r", libver="latest") as fin:
            data["amino_acids"] = fin["amino_acids"][...]

    # Create new dsets from concatenated dataset
    fout = h5py.File(output_name, "w", libver="latest")
    for field, concat_dset in data.items():

        shape = concat_dset.shape
        chunkshape = (1,) + shape[1:]
        # Create dataset
        if concat_dset.dtype != np.object:
            if np.any(np.isnan(concat_dset)):
                raise ValueError("NaN detected in concat_dset.")
            dtype = concat_dset.dtype
        else:
            if field == "contact_map":  # contact_map is integer valued
                dtype = h5py.vlen_dtype(np.int16)
            else:
                dtype = h5py.vlen_dtype(np.float32)

        dset = fout.create_dataset(field, shape, chunks=chunkshape, dtype=dtype)
        # write data
        dset[...] = concat_dset[...]

    # Clean up
    fout.flush()
    fout.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_files_glob",
        help="Path sring to glob HDF5 files.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_name",
        help="Output HDF5 file to write.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    input_file_names = glob.glob(args.input_files_glob)
    print("Concatenating:", input_file_names)
    concatenate_h5(input_file_names, args.output_name)
