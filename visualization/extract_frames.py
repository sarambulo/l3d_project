import argparse
import imageio
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from GIF files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .gif files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for extracted frames"
    )
    args = parser.parse_args()

    # Paths
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    # Validate input directory
    assert input_path.exists(), f"Input directory {input_path} not found"
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .gif files
    gif_files = sorted(input_path.glob("*.gif"))
    print(f"Found {len(gif_files)} .gif files in {input_path}")

    # Extract frames from each GIF
    for gif_file in tqdm(gif_files, desc="Extracting frames"):
        # Read GIF
        gif_data = imageio.mimread(str(gif_file))
        
        # Extract frames (1st frame is index 0, 5th frame is index 4)
        frames_to_extract = [0, 4]
        
        for frame_idx in frames_to_extract:
            if frame_idx < len(gif_data):
                frame = gif_data[frame_idx]
                
                # Save frame as JPEG
                output_filename = output_path / f"{gif_file.stem}_frame_{frame_idx}.jpeg"
                imageio.imwrite(str(output_filename), frame)

    print(f"Extracted frames to {output_path}")


if __name__ == "__main__":
    main()
