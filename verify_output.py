import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths
input_file = "/home/jrenner/local/lbl/holoscan/sparse_frames.h5"
output_file = "/home/jrenner/local/lbl/holoscan/sparse_frames_out.h5"

def verify_and_plot():
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return
    if not os.path.exists(output_file):
        print(f"Error: Output file not found at {output_file}")
        return

    try:
        with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'r') as f_out:
            print("Input keys:", list(f_in.keys()))
            print("Output keys:", list(f_out.keys()))

            # Assuming dataset names based on yaml config
            ds_in_name = "frames"
            ds_out_name = "processed"

            if ds_in_name not in f_in:
                print(f"Error: Dataset '{ds_in_name}' not found in input file.")
                return
            if ds_out_name not in f_out:
                print(f"Error: Dataset '{ds_out_name}' not found in output file.")
                return

            data_in = f_in[ds_in_name]
            data_out = f_out[ds_out_name]

            print(f"Input shape: {data_in.shape}")
            print(f"Output shape: {data_out.shape}")

            num_frames = min(data_in.shape[0], data_out.shape[0], 3) # Plot up to 3 frames

            fig, axes = plt.subplots(num_frames, 2, figsize=(10, 5 * num_frames))
            if num_frames == 1:
                axes = np.expand_dims(axes, axis=0)

            for i in range(num_frames):
                # Input frame
                frame_in = data_in[i]
                # If 3D (1, H, W) or (H, W), handle it. Assuming (H, W) or (1, H, W)
                if frame_in.ndim == 3:
                    frame_in = frame_in[0]
                
                ax_in = axes[i, 0]
                im_in = ax_in.imshow(frame_in, cmap='viridis')
                ax_in.set_title(f"Input Frame {i}")
                plt.colorbar(im_in, ax=ax_in)

                # Output frame
                frame_out = data_out[i]
                if frame_out.ndim == 3: # (1, H, W) or (C, H, W)
                    frame_out = frame_out[0] # Take first channel
                
                ax_out = axes[i, 1]
                im_out = ax_out.imshow(frame_out, cmap='viridis')
                ax_out.set_title(f"Output Frame {i} (Convolved)")
                plt.colorbar(im_out, ax=ax_out)

            plt.tight_layout()
            output_plot_path = "frame_comparison.png"
            plt.savefig(output_plot_path)
            print(f"Comparison plot saved to {output_plot_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    verify_and_plot()
