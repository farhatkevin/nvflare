import argparse
import os

def check_header(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"--- Checking header of file: {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(6)
            if magic == b'\x93NUMPY':
                print("File starts with the correct NumPy magic string: b'\\x93NUMPY'")
                version_major = f.read(1)
                version_minor = f.read(1)
                print(f"NPY Format Version: {ord(version_major)}.{ord(version_minor)}")
                if ord(version_major) == 1:
                    header_len_bytes = f.read(2)
                    header_len = int.from_bytes(header_len_bytes, byteorder='little')
                elif ord(version_major) >= 2:
                    header_len_bytes = f.read(4)
                    header_len = int.from_bytes(header_len_bytes, byteorder='little')
                else:
                    print("Unknown NPY version for header length.")
                    return
                print(f"Declared header length: {header_len}")
                header_str = f.read(header_len).decode('utf-8', errors='ignore')
                print(f"Header content (first 500 chars):\n{header_str[:500]}")
                if "'dtype': 'O'" in header_str or "'descr': '|O'" in header_str :
                    print("\n*** Header indicates an object dtype ('O'). This means allow_pickle=True is likely needed for np.load. ***")
                else:
                    print("\nHeader does not immediately indicate an object dtype. Dtype info:", header_str)


            else:
                print(f"File DOES NOT start with the NumPy magic string. Actual start: {magic}")
                print("This file is likely not a standard .npy file.")

    except Exception as e:
        print(f"An error occurred while reading the file header: {e}")

    print("\n--- Header Check Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the header of a .npy file.")
    parser.add_argument("file_path", type=str, help="Path to the .npy file.")
    args = parser.parse_args()
    check_header(args.file_path)