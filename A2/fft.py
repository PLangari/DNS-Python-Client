import argparse

def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Fast Fourier Transform and Applications')

    # Define the command-line arguments
    parser.add_argument('-m', '--mode', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Mode: 1 - Display FFT, 2 - Denoise, 3 - Compress and plot, 4 - Plot runtime graphs')
    parser.add_argument('-i', '--image', type=str, default='moonlanding.png',
                        help='Filename of the image to be processed')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the arguments
    mode = args.mode
    image_filename = args.image

    # Perform actions based on the selected mode
    if mode == 1:
        print(f"Displaying FFT of the image: {image_filename}")
        # Call the function to display FFT
        # display_fft(image_filename)
    elif mode == 2:
        print(f"Denoising the image: {image_filename}")
        # Call the function for denoising
        # denoise_image(image_filename)
    elif mode == 3:
        print(f"Compressing and plotting the image: {image_filename}")
        # Call the function for compression and plotting
        # compress_and_plot(image_filename)
    elif mode == 4:
        print("Plotting runtime graphs")
        # Call the function for plotting runtime graphs
        # plot_runtime_graphs()
    else:
        print("Invalid mode. Please choose a valid mode: 1, 2, 3, or 4.")


if __name__ == "__main__":
    parse_arguments()