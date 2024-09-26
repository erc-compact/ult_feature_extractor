import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq

# Define sine function for curve_fit
def sin_func(x, amp, freq, phase, bg):
    return amp * np.sin(2 * np.pi * freq * x + phase) + bg

# Define residuals function for leastsq method
def residuals(paras, x, y, amp, bg):
    freq, phase = paras
    return y - (amp * np.sin(2 * np.pi * freq * x + phase) + bg)

# Define evaluation function for leastsq method
def evaluate(x, paras, amp, bg):
    freq, phase = paras
    return amp * np.sin(2 * np.pi * freq * x + phase) + bg

# Function to fit using curve_fit
def sin_fit_curve_fit(xData, yData, debug=False):
    background_guess = (max(yData) + min(yData)) / 2
    amplitude_guess = (max(yData) - min(yData)) / 2 
    
    frequency_guess = 1
    phase_guess = 0

    # Perform curve_fit
    popt, _ = curve_fit(sin_func, xData, yData, p0=[amplitude_guess, frequency_guess, phase_guess, background_guess])
    yFit = sin_func(xData, *popt)
    chi_square = np.sum((yData - yFit) ** 2)
    
    if debug:
        print('first:' ,amplitude_guess, frequency_guess,  phase_guess, background_guess)
        print(popt)
        plt.plot(xData, yData, 'o', label='Original Data')
        plt.plot(xData, yFit, label='Curve Fit Sine')
        plt.title("Sine Fit - curve_fit")
        plt.legend()
        plt.savefig('curve_fit.png')  # Save the plot to a file
        plt.close()  # Close the plot to avoid displaying it
    
    return chi_square, popt

# Function to fit using leastsq
def sin_fit_leastsq(xData, yData, debug=False):
    # Estimate initial amplitude and background from the data
    amplitude = (max(yData) - min(yData)) / 2
    background = (max(yData) + min(yData)) / 2
    
    # Estimate frequency from number of maxima (or assume initial guess of 1)
    frequency_guess = 1
    phi0 = 0  # Initial phase guess

    # Perform leastsq fit for frequency and phase
    parameters = [frequency_guess, phi0]
    leastSquaresParameters, _, _, _, _ = leastsq(residuals, parameters, args=(xData, yData, amplitude, background), full_output=True)
    yFit = evaluate(xData, leastSquaresParameters, amplitude, background)
    chi_square = np.sum((yData - yFit) ** 2)
    
    if debug:
        plt.plot(xData, yData, 'o', label='Original Data')
        plt.plot(xData, yFit, label='Leastsq Sine')
        plt.title("Sine Fit - leastsq")
        plt.legend()
        plt.savefig('leastsq.png')  # Save the plot to a file
        plt.close()  # Close the plot to avoid displaying it
    
    return chi_square, leastSquaresParameters, amplitude, background

# Function to fit with curve_fit and fixed parameters
def sin_fit_curve_fit_amp_bg(xData, yData, debug=False):
    def sin_func_fixed(x, freq, phase):
        return sin_func(x, amplitude, freq, phase, background)
    
    # Estimate amplitude and background
    def estimate_amp_bg(xData, yData):
        amplitude = (max(yData) - min(yData)) / 2
        background = (max(yData) + min(yData)) / 2
        return amplitude, background
    
    amp_bg_params = estimate_amp_bg(xData, yData)
    amplitude, background = amp_bg_params
    
    # Fit the amplitude and background
    def sin_func_initial(x, amp, bg):
        return sin_func(x, amp, 0, 0, bg)
    
    popt_amp_bg, _ = curve_fit(sin_func_initial, xData, yData, p0=[amplitude, background])
    amplitude, background = popt_amp_bg
    
    # Now use fixed amplitude and background to fit frequency and phase
    def sin_func_fixed_params(x, freq, phase):
        return sin_func(x, amplitude, freq, phase, background)
    
    initial_freq = 1
    initial_phase = 0
    popt_freq_phase, _ = curve_fit(sin_func_fixed_params, xData, yData, p0=[initial_freq, initial_phase])
    
    yFit = sin_func_fixed_params(xData, *popt_freq_phase)
    chi_square = np.sum((yData - yFit) ** 2)
    
    if debug:
        plt.plot(xData, yData, 'o', label='Original Data')
        plt.plot(xData, yFit, label='Curve Fit with Fixed Frequency and Phase')
        plt.title("Sine Fit - curve_fit with Fixed Frequency and Phase")
        plt.legend()
        plt.savefig('curve_fit_fixed_params.png')  # Save the plot to a file
        plt.close()  # Close the plot to avoid displaying it
    
    return chi_square, popt_freq_phase, amplitude, background

# Function to compare the three fitting methods
def compare_sine_fits(xData, yData, debug=False):
    chi_square_curve_fit, popt_curve_fit = sin_fit_curve_fit(xData, yData, debug)
    chi_square_leastsq, popt_leastsq, _, _ = sin_fit_leastsq(xData, yData, debug)
    chi_square_fixed_params, popt_fixed_params, amp, bg = sin_fit_curve_fit_amp_bg(xData, yData, debug)
    
    if debug:
        # Plot the original data and all fits
        plt.plot(xData, yData, 'o', label='Original Data')
        plt.plot(xData, sin_func(xData, *popt_curve_fit), label='Curve Fit Sine')
        plt.plot(xData, evaluate(xData, popt_leastsq, (max(yData) - min(yData)) / 2, (max(yData) + min(yData)) / 2), label='Leastsq Sine')
        plt.plot(xData, sin_func(xData, *(list(popt_fixed_params) + [amp, bg])), label='Curve Fit Fixed Params')
        plt.title("Sine Fit Comparison")
        plt.legend()
        plt.savefig('sine_fit_comparison.png')  # Save the combined plot to a file
        plt.close()  # Close the plot to avoid displaying it
    
    # Print comparison of chi-squares and parameters using format
    print("curve_fit: chi-square = {}, parameters = {}".format(chi_square_curve_fit, popt_curve_fit))
    print("leastsq: chi-square = {}, parameters (frequency, phase) = {}".format(chi_square_leastsq, popt_leastsq))
    print("curve_fit with fixed params: chi-square = {}, parameters (frequency, phase) = {}, amplitude = {}, background = {}".format(
        chi_square_fixed_params, popt_fixed_params, amp, bg))

# Generate some sample data for testing
np.random.seed(0)
xData = np.linspace(0, 10, 100)
true_amp = 2
true_freq = 0.5
true_phase = 1
true_bg = 1
yData = true_amp * np.sin(2 * np.pi * true_freq * xData + true_phase) + true_bg + 0.2 * np.random.normal(size=len(xData))

# Run the comparison with debug mode on
compare_sine_fits(xData, yData, debug=True)
