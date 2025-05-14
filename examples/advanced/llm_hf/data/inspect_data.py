import numpy as np
import argparse
import os
import pickle

def inspect_npy_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Inspecting file: {file_path} ---")

    # 1. Try loading with np.load, allow_pickle=True (as in your script)
    print("\nAttempt 1: np.load(file_path, allow_pickle=True)")
    try:
        data_np_pickle_true = np.load(file_path, allow_pickle=True)
        print("Successfully loaded with allow_pickle=True.")
        print(f"  Data type: {type(data_np_pickle_true)}")
        print(f"  NumPy array dtype: {data_np_pickle_true.dtype if isinstance(data_np_pickle_true, np.ndarray) else 'N/A'}")
        print(f"  Shape: {data_np_pickle_true.shape if isinstance(data_np_pickle_true, np.ndarray) else 'N/A'}")
        if isinstance(data_np_pickle_true, np.ndarray) and data_np_pickle_true.size > 0:
            print(f"  First element type: {type(data_np_pickle_true.item(0) if data_np_pickle_true.ndim == 0 else data_np_pickle_true[0])}")
            print(f"  First element sample: {data_np_pickle_true.item(0) if data_np_pickle_true.ndim == 0 else data_np_pickle_true[0][:100] if isinstance(data_np_pickle_true[0], (str, list, bytes)) else data_np_pickle_true[0]}")
        elif not isinstance(data_np_pickle_true, np.ndarray):
             print(f"  First element type (if list/tuple): {type(data_np_pickle_true[0]) if len(data_np_pickle_true) > 0 else 'N/A'}")
             print(f"  First element sample (if list/tuple): {data_np_pickle_true[0][:100] if len(data_np_pickle_true) > 0 and isinstance(data_np_pickle_true[0], (str, list, bytes)) else (data_np_pickle_true[0] if len(data_np_pickle_true) > 0 else 'N/A') }")


    except Exception as e:
        print(f"Failed to load with allow_pickle=True: {e}")

    # 2. Try loading with np.load, allow_pickle=False (the default for .npy)
    print("\nAttempt 2: np.load(file_path, allow_pickle=False)")
    try:
        data_np_pickle_false = np.load(file_path, allow_pickle=False)
        print("Successfully loaded with allow_pickle=False.")
        print(f"  Data type: {type(data_np_pickle_false)}")
        print(f"  NumPy array dtype: {data_np_pickle_false.dtype}")
        print(f"  Shape: {data_np_pickle_false.shape}")
        if data_np_pickle_false.size > 0:
            print(f"  First element type: {type(data_np_pickle_false.item(0) if data_np_pickle_false.ndim == 0 else data_np_pickle_false[0])}")
            print(f"  First element sample: {data_np_pickle_false.item(0) if data_np_pickle_false.ndim == 0 else data_np_pickle_false[0][:100] if isinstance(data_np_pickle_false[0], (str, list, bytes)) else data_np_pickle_false[0]}")
    except Exception as e:
        print(f"Failed to load with allow_pickle=False: {e}")
        if "Object arrays are not S렇pported" in str(e) or "allow_pickle=True" in str(e):
            print("  This suggests it's a NumPy object array saved without pickle, requiring allow_pickle=True for loading.")
        elif "Failed to interpret file" in str(e) or "not a richtige NPY file" in str(e).lower(): # German error message
             print("  This strongly suggests it's not a standard .npy file.")


    # 3. Try loading as a raw pickle file (if it has .npy extension but is actually .pkl)
    print("\nAttempt 3: Loading as a raw pickle file (pickle.load)")
    try:
        with open(file_path, 'rb') as f:
            data_pickle = pickle.load(f)
        print("Successfully loaded as a raw pickle file.")
        print(f"  Data type: {type(data_pickle)}")
        if hasattr(data_pickle, 'shape'):
            print(f"  Shape (if any): {data_pickle.shape}")
        if hasattr(data_pickle, 'dtype'):
            print(f"  Dtype (if any): {data_pickle.dtype}")
        if isinstance(data_pickle, (list, tuple)) and len(data_pickle) > 0:
            print(f"  Length (if list/tuple): {len(data_pickle)}")
            print(f"  First element type: {type(data_pickle[0])}")
            print(f"  First element sample: {str(data_pickle[0])[:200]}") # Print first 200 chars of string representation
        elif isinstance(data_pickle, np.ndarray) and data_pickle.size > 0:
            print(f"  First element type: {type(data_pickle.item(0) if data_pickle.ndim == 0 else data_pickle[0])}")
            print(f"  First element sample: {data_pickle.item(0) if data_pickle.ndim == 0 else data_pickle[0][:100] if isinstance(data_pickle[0], (str, list, bytes)) else data_pickle[0]}")

    except pickle.UnpicklingError as e:
        print(f"Failed to load as a raw pickle file (UnpicklingError): {e}")
        print("  This suggests it's not a standard pickle file.")
    except Exception as e:
        print(f"Failed to load as a raw pickle file (Other Error): {e}")

    print("\n--- Inspection Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a .npy (or suspected .npy) file.")
    parser.add_argument("file_path", type=str, help="Path to the .npy file to inspect.")
    args = parser.parse_args()

    inspect_npy_file(args.file_path)