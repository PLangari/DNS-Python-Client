import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

def fft(x):
    """
    Compute the 1D Fast Fourier Transform (FFT) of input signal x.
    
    Parameters:
        x (np.ndarray): Input signal.
    
    Returns:
        np.ndarray: FFT of the input signal.
    """
    N = len(x)
    
    # Base case: if the input size is 1, return the input itself
    if N == 1:
        return x
    
    # Split the input into even and odd indices
    even_indices = x[::2]
    odd_indices = x[1::2]
    
    # Recursively compute FFT for even and odd indices
    fft_even = fft(even_indices)
    fft_odd = fft(odd_indices)
    
    # Combine results using FFT butterfly operation
    t = np.exp(-2j * np.pi * np.arange(N) / N)
    fft_combined = np.concatenate([fft_even + t[:N//2] * fft_odd, fft_even + t[N//2:] * fft_odd])
    
    return fft_combined

def fft2d(x):
    """
    Compute the 2D Fast Fourier Transform (FFT) of input signal x.
    
    Parameters:
        x (np.ndarray): Input signal (2D array).
    
    Returns:
        np.ndarray: 2D FFT of the input signal.
    """
    # Apply 1D FFT to rows
    rows_fft = np.apply_along_axis(fft, axis=1, arr=x)
    
    # Apply 1D FFT to columns
    cols_fft = np.apply_along_axis(fft, axis=0, arr=rows_fft)

    fft_result_shifted = np.fft.fftshift(cols_fft)
    
    return fft_result_shifted

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

def test_signal():
    # Test signal
    x1 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    x2 = np.random.randint(0, 100, 512)
    x3 = np.random.randint(0, 100, 256)
    x4 = np.random.randint(0, 100, 128)

    for x in [x1, x2, x3, x4]:
        # Calculate FFT using your implementation
        your_fft_result = fft_cooley_tukey(x)

        # Calculate FFT using NumPy
        numpy_fft_result = np.fft.fft(x)

        # Compare the results
        # print("Your FFT Result:", your_fft_result)
        # print("NumPy FFT Result:", numpy_fft_result)

        # Check if the results are close within a certain tolerance
        tolerance = 1e-10
        if np.allclose(your_fft_result, numpy_fft_result, rtol=tolerance, atol=tolerance):
            print("Results are close. Your FFT implementation seems correct.")
        else:
            print("Results differ. Please double-check your FFT implementation.")

def test_images():
   # Read an image using cv2
    image_path = 'flash.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Unable to load the image.")
    else:
        # Apply 2D FFT to the image
        fft_result_2d = fft2d(image)

        reference_fft_result = np.fft.fft2(image)

        # Check if the results are close within a certain tolerance
        tolerance = 1e-10
        if np.allclose(fft_result_2d, reference_fft_result, rtol=tolerance, atol=tolerance):
            print("Results are close. Your 2D FFT implementation seems correct.")
        else:
            print("Results differ. Please double-check your 2D FFT implementation.")

        # Display the original and 2D FFT images using matplotlib
        plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(1, 2, 2), plt.imshow(np.log(1 + np.abs(fft_result_2d)), cmap='gray')
        plt.title('2D FFT of Image'), plt.xticks([]), plt.yticks([])

        plt.show()

def ref_test():
    # Load the image
    image = plt.imread('flash.png')  # Replace 'path/to/your/image.jpg' with the actual path to your image

    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    # Perform 2D FFT
    fft_result = fft2(image)

    # Shift zero frequency components to the center
    fft_result_shifted = fftshift(fft_result)

    # Calculate magnitude spectrum (log-scaled for better visualization)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1)

    # Display the original image and its magnitude spectrum
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum (2D FFT)')
    plt.show()


test_signal()