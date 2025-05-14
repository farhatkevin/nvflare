#!/usr/bin/env python3
"""
simple_s3_downloader.py manifest1.txt manifest2.txt --dest ./data/
"""
import argparse
import subprocess
import pathlib
import sys

def main():
    ap = argparse.ArgumentParser(
        description="Simplified script to download files from S3 based on a manifest. "
                    "Each manifest line should be 'name,relative/s3/path/to/file.ext'. "
                    "The 'name' part is ignored.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("manifests", nargs="+",
                    help="Manifest file(s) (*.txt).")
    ap.add_argument("--dest", required=True,
                    help="Base local directory. Files from 'manifest.txt' will go into '--dest/manifest/'.")
    ap.add_argument("--s3-prefix", default="s3://ai2-llm/",
                    help="S3 prefix to prepend to relative paths from the manifest (default: s3://ai2-llm/).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands but do not execute them.")
    args = ap.parse_args()

    base_dest_root = pathlib.Path(args.dest).expanduser().resolve()
    base_dest_root.mkdir(parents=True, exist_ok=True)
    print(f"INFO: Base destination directory: {base_dest_root}")

    # Ensure the S3 prefix is well-formed for simple concatenation and starts with s3://
    s3_prefix_clean = args.s3_prefix.rstrip('/')
    if not s3_prefix_clean.startswith("s3://"):
        print(f"ERROR: --s3-prefix '{args.s3_prefix}' must start with 's3://'. Exiting.", file=sys.stderr)
        sys.exit(1)

    for manifest_filepath_str in args.manifests:
        manifest_filepath = pathlib.Path(manifest_filepath_str)
        if not manifest_filepath.is_file():
            print(f"WARNING: Manifest file not found, skipping: {manifest_filepath_str}", file=sys.stderr)
            continue

        # Subdirectory for this manifest's files (e.g., .../data/math for math.txt)
        manifest_stem = manifest_filepath.stem
        target_subdir = base_dest_root / manifest_stem
        print(f"DEBUG: For manifest '{manifest_filepath_str}', manifest_stem is '{manifest_stem}', target_subdir is '{target_subdir}'") # <--- ADD THIS LINE

        target_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing manifest '{manifest_filepath_str}', target directory: '{target_subdir}'")

        with open(manifest_filepath, 'r') as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    # print(f"  Line {i}: Skipping empty or comment line.") # Can be noisy
                    continue

                try:
                    # Disregard the part before the comma
                    _name_ignored, relative_s3_path = line.split(",", 1)
                    relative_s3_path = relative_s3_path.strip()
                except ValueError:
                    print(f"  Line {i}: SKIPPING malformed line (no comma found): '{line}'", file=sys.stderr)
                    continue

                if not relative_s3_path:
                    print(f"  Line {i}: SKIPPING line with empty path after comma: '{line}'", file=sys.stderr)
                    continue
                
                # Construct the full S3 URI by simple concatenation
                # Example: s3_prefix_clean = "s3://ai2-llm"
                #          relative_s3_path = "preprocessed/.../file.npy" (or can be "/preprocessed/.../file.npy")
                #          full_s3_uri will be "s3://ai2-llm/preprocessed/.../file.npy"
                full_s3_uri = f"{s3_prefix_clean}/{relative_s3_path.lstrip('/')}"

                # Extract just the filename for the local destination
                filename_only = pathlib.Path(relative_s3_path).name
                if not filename_only or filename_only == "." or filename_only == "..": # Ensure a valid filename
                    print(f"  Line {i}: SKIPPING line, could not extract a valid filename from S3 path: '{relative_s3_path}' (extracted: '{filename_only}')", file=sys.stderr)
                    continue
                
                local_destination_path = target_subdir / filename_only

                # print(f"  Line {i}: Plan to download '{full_s3_uri}' to '{local_destination_path}'") # Can be noisy

                if local_destination_path.exists():
                    print(f"  INFO: Already exists, skipping: {local_destination_path}")
                    continue

                cmd = ["aws", "s3", "cp", full_s3_uri, str(local_destination_path)]
                
                print(f"  COMMAND: {' '.join(cmd)}")

                if not args.dry_run:
                    try:
                        # print(f"    EXECUTING...") # Can be noisy
                        # For aws s3 cp, progress is often on stderr. stdout might be empty on success.
                        process = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
                        # if process.stdout: print(f"      STDOUT: {process.stdout.strip()}")
                        if process.stderr: # Print S3's output (often progress or final summary)
                             print(f"      S3 Op Output: {process.stderr.strip()}", file=sys.stderr) # aws s3 cp messages go to stderr
                        print(f"    SUCCESS: Downloaded '{filename_only}'")
                    except subprocess.CalledProcessError as e:
                        print(f"    ERROR: Command failed for '{full_s3_uri}'.", file=sys.stderr)
                        print(f"      Return code: {e.returncode}", file=sys.stderr)
                        if e.stdout: print(f"      Stdout: {e.stdout.strip()}", file=sys.stderr)
                        if e.stderr: print(f"      Stderr: {e.stderr.strip()}", file=sys.stderr)
                        # You might want to decide whether to continue or exit on error
                        # print("Continuing with next file...") 
                        # sys.exit(1) # Uncomment to stop on first error
                    except FileNotFoundError:
                        print(f"    ERROR: AWS CLI command ('aws') not found. Is it installed and in your system's PATH?", file=sys.stderr)
                        sys.exit("AWS CLI not found. Please install it and ensure it's in your PATH.")
                    except Exception as e_py:
                        print(f"    ERROR: A Python error occurred during subprocess execution: {e_py}", file=sys.stderr)
                        # sys.exit("Unexpected Python error.") # Uncomment to stop on Python error

    print("\nAll manifest processing complete.")

if __name__ == "__main__":
    main()