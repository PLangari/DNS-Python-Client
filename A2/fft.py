import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

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
    return (mode, image_filename)

def resize(image):
    # Calculate the nearest power of 2 dimensions
    height, width = image.shape[:2]
    new_height = 2 ** int(np.ceil(np.log2(height)))
    new_width = 2 ** int(np.ceil(np.log2(width)))

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def naive_dft(x):
    """
    Compute the 1D Discrete Fourier Transform (DFT) of input signal x using the naive approach.
    
    Parameters:
        x (np.ndarray): Input signal.
    
    Returns:
        np.ndarray: DFT of the input signal.
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    omega = np.exp(-2j * np.pi * k * n / N)
    return np.dot(omega, x)

def fft_cooley_tukey(x):
    """
    Compute the 1D Fast Fourier Transform (FFT) of input signal x using the Cooley-Tukey algorithm
    with the naive DFT applied at the end.
    
    Parameters:
        x (np.ndarray): Input signal.
    
    Returns:
        np.ndarray: FFT of the input signal.
    """
    N = len(x)
    
    # Base case: if the input size is 1, return the input itself
    if N <= 32:
        return naive_dft(x)
    
    # Split the input into even and odd indices
    even_indices = x[::2]
    odd_indices = x[1::2]
    
    # Recursively compute FFT for even and odd indices
    fft_even = fft_cooley_tukey(even_indices)
    fft_odd = fft_cooley_tukey(odd_indices)
    
    # Combine results using naive DFT instead of butterfly operation
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    fft_combined = np.concatenate([fft_even + t[:N//2] * fft_odd, fft_even + t[N//2:] * fft_odd])
    
    return fft_combined

def fft2d(x, naive=False):
    """
    Compute the 2D Fast Fourier Transform (FFT) of input signal x.
    
    Parameters:
        x (np.ndarray): Input signal (2D array).
    
    Returns:
        np.ndarray: 2D FFT of the input signal.
    """
    if not naive:
        # Apply 1D FFT to rows
        rows_fft = np.apply_along_axis(fft_cooley_tukey, axis=1, arr=x)
        
        # Apply 1D FFT to columns
        cols_fft = np.apply_along_axis(fft_cooley_tukey, axis=0, arr=rows_fft)

        fft_result_shifted = np.fft.fftshift(cols_fft)
    else:
        # Apply 1D FFT to rows
        rows_fft = np.apply_along_axis(naive_dft, axis=1, arr=x)
        
        # Apply 1D FFT to columns
        cols_fft = np.apply_along_axis(naive_dft, axis=0, arr=rows_fft)

        fft_result_shifted = np.fft.fftshift(cols_fft)
    
    return fft_result_shifted

def display_fft(org_img, trfm_img):
    # Magnitude spectrum
    magnitude_spectrum_org = np.abs(trfm_img)

    # Plot original image and its Fourier Transform
    plt.figure(figsize=(10, 4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(org_img, cmap='gray')
    plt.title('Original Image')

    # Fourier Transform with log scale
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum_org, cmap='gray', norm=LogNorm())
    plt.title('Fourier Transform (Log Scaled)')

    plt.tight_layout()
    plt.show()

def apply_compression(fft_result, compression_level):
    """
    Apply compression to the FFT result.

    Parameters:
        fft_result (np.ndarray): The 2D FFT result.
        compression_level (float): The percentage of coefficients to keep.

    Returns:
        np.ndarray: Compressed FFT result.
    """
    # Flatten the FFT coefficients
    flat_fft = np.abs(fft_result).flatten()

    # Sort the coefficients in descending order
    sorted_indices = np.argsort(flat_fft)[::-1]

    # Determine the threshold index based on compression level
    threshold_index = int(compression_level * len(sorted_indices))

    # Set coefficients beyond the threshold to zero
    flat_fft[sorted_indices[threshold_index:]] = 0

    # Reshape the flattened array back to the original shape
    compressed_fft = flat_fft.reshape(fft_result.shape)

    return compressed_fft

def apply_compression(fft_result, compression_level):
    """
    Apply compression to the FFT result.

    Parameters:
        fft_result (np.ndarray): The 2D FFT result.
        compression_level (float): The percentage of coefficients to keep.

    Returns:
        np.ndarray: Compressed FFT result.
    """
    # Flatten the FFT coefficients
    flat_fft = np.abs(fft_result).flatten()

    # Sort the coefficients in descending order
    sorted_indices = np.argsort(flat_fft)[::-1]

    # Determine the threshold index based on compression level
    threshold_index = int(compression_level * len(sorted_indices))

    # Set coefficients beyond the threshold to zero
    flat_fft[sorted_indices[threshold_index:]] = 0

    # Reshape the flattened array back to the original shape
    compressed_fft = flat_fft.reshape(fft_result.shape)

    return compressed_fft

def runtime_comparison(mode=4, min_size_exp=5, max_size_exp=10, num_trials=10):
    """
    Perform runtime complexity comparison between FFT and naive DFT for different problem sizes.

    Parameters:
        mode (int): The mode of operation (4 for runtime comparison).
        min_size_exp (int): The minimum exponent for the problem size (2**min_size_exp).
        max_size_exp (int): The maximum exponent for the problem size (2**max_size_exp).
        num_trials (int): Number of trials for each problem size.

    Returns:
        None
    """
    problem_sizes = [2**i for i in range(min_size_exp, max_size_exp + 1)]

    fft_runtimes = []
    naive_dft_runtimes = []

    for size in problem_sizes:
        # Generate a random 2D array of size x size
        test_input = np.random.rand(size, size)

        fft_times = []
        naive_dft_times = []

        for _ in range(num_trials):
            # Measure runtime for FFT
            start_time = time.time()
            fft_result = fft2d(test_input)
            end_time = time.time()
            fft_times.append(end_time - start_time)

            # Measure runtime for naive DFT
            start_time = time.time()
            naive_dft_result = fft2d(test_input, True)
            end_time = time.time()
            naive_dft_times.append(end_time - start_time)

        # Calculate mean and standard deviation for each algorithm
        mean_fft_time = np.mean(fft_times)
        mean_naive_dft_time = np.mean(naive_dft_times)
        std_fft_time = np.std(fft_times)
        std_naive_dft_time = np.std(naive_dft_times)

        # Record mean runtime for each algorithm
        fft_runtimes.append(mean_fft_time)
        naive_dft_runtimes.append(mean_naive_dft_time)

        # Print information
        print(f"Size: {size}, FFT Mean Time: {mean_fft_time:.6f} seconds, FFT Std Dev: {std_fft_time:.6f} seconds")
        print(f"Size: {size}, Naive DFT Mean Time: {mean_naive_dft_time:.6f} seconds, Naive DFT Std Dev: {std_naive_dft_time:.6f} seconds")

    # Create a plot
    plt.plot(problem_sizes, fft_runtimes, label='FFT', marker='o')
    plt.plot(problem_sizes, naive_dft_runtimes, label='Naive DFT', marker='o')
    plt.errorbar(problem_sizes, fft_runtimes, yerr=std_fft_time, fmt='o', capsize=5, label='FFT')
    plt.errorbar(problem_sizes, naive_dft_runtimes, yerr=std_naive_dft_time, fmt='o', capsize=5, label='Naive DFT')

    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Problem Size (log scale)')
    plt.ylabel('Mean Runtime (log scale)')
    plt.title('Runtime Complexity Comparison: FFT vs Naive DFT')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parse the command-line arguments
    mode, file_name = parse_arguments() 

    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    r_img = resize(img)

    # Perform actions based on the selected mode
    if mode == 1:
        print(f"Displaying FFT of the image: {file_name}")
        # Call the function to display FFT
        fft_transform = fft2d(r_img)
        display_fft(img, fft_transform)
    elif mode == 2:
        print(f"Denoising the image: {file_name}")
        # Call the function for denoising
    elif mode == 3:
        print(f"Compressing and plotting the image: {file_name}")
        # Call the function for compression and plotting
        fft_transform = fft2d(r_img)
        compression_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.999]
        plt.figure(figsize=(12,8))
        for i, complresion_level in enumerate(compression_levels, 1):
            compressed_fft = apply_compression(fft_transform, complresion_level)

            # Apply inverse FFT
            reconstructed_img = None

            plt.subplot(2,3, i)
            plt.imshow(np.abs(reconstructed_img), cmap='gray')
            plt.title(f"Compression Level: {complresion_level * 100}")
            plt.xticks([]), plt.yticks([])

            # Print information

        fft_transform = fft2d(r_img)
        compression_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.999]
        plt.figure(figsize=(12,8))
        for i, complresion_level in enumerate(compression_levels, 1):
            compressed_fft = apply_compression(fft_transform, complresion_level)

            # Apply inverse FFT
            reconstructed_img = None

            plt.subplot(2,3, i)
            plt.imshow(np.abs(reconstructed_img), cmap='gray')
            plt.title(f"Compression Level: {complresion_level * 100}")
            plt.xticks([]), plt.yticks([])

            # Print information

    elif mode == 4:
        print("Plotting runtime graphs")
        # Call the function for plotting runtime graphs
        runtime_comparison()
    else:
        print("Invalid mode. Please choose a valid mode: 1, 2, 3, or 4.")

