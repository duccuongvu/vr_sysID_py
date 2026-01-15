from scipy import signal
import numpy as np


def filter_by_time_range(q, dq, tau, time_data, t_start, t_end):
    """Filter data to keep only samples within the specified time range."""
    # Find indices within time range
    mask = (time_data >= t_start) & (time_data <= t_end)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        raise ValueError(f"No data found in time range [{t_start}, {t_end}] seconds")
    
    # Filter all data arrays
    q_filtered = q[:, indices]
    dq_filtered = dq[:, indices]
    tau_filtered = tau[:, indices]
    time_filtered = time_data[indices]
    
    print(f"\nFiltering data by time range:")
    print(f"  Original: {len(time_data)} samples, time range [{time_data[0]:.3f}, {time_data[-1]:.3f}] seconds")
    print(f"  Filtered: {len(indices)} samples, time range [{time_data[indices[0]]:.3f}, {time_data[indices[-1]]:.3f}] seconds")
    print(f"  Removed: {len(time_data) - len(indices)} samples outside [{t_start}, {t_end}] seconds")
    
    return q_filtered, dq_filtered, tau_filtered, time_filtered


def butterworth_filter(data, cutoff_freq, sample_rate, order=2):
    """Apply zero-phase Butterworth low-pass filter."""
    nyquist = sample_rate / 2.0
    if cutoff_freq >= nyquist:
        print(f"Warning: Cutoff frequency >= Nyquist, reducing to 0.9 * Nyquist")
        cutoff_freq = 0.9 * nyquist
    
    # Design filter
    sos = signal.butter(order, cutoff_freq / nyquist, output='sos')
    
    # Apply forward-backward filter for zero phase
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    
    return filtered


def filter_velocity_data(dq, dt, cutoff_freq=50.0):
    """Filter velocity data using Butterworth filter."""
    sample_rate = 1.0 / dt
    print(f"Filtering velocity data (cutoff: {cutoff_freq:.2f} Hz, sample rate: {sample_rate:.2f} Hz)")
    dq_filt = butterworth_filter(dq, cutoff_freq, sample_rate, order=2)
    return dq_filt


def compute_acceleration(dq_filt, dt):
    """Compute acceleration from filtered velocity using numerical differentiation."""
    print("Computing acceleration from filtered velocity...")
    ddq = np.zeros_like(dq_filt)
    
    # Central difference for interior points
    ddq[:, 1:-1] = (dq_filt[:, 2:] - dq_filt[:, :-2]) / (2.0 * dt)
    
    # Forward difference for first point
    if dq_filt.shape[1] > 1:
        ddq[:, 0] = (dq_filt[:, 1] - dq_filt[:, 0]) / dt
    
    # Backward difference for last point
    if dq_filt.shape[1] > 1:
        ddq[:, -1] = (dq_filt[:, -1] - dq_filt[:, -2]) / dt
    
    return ddq


def filter_acceleration_data(ddq, dt, cutoff_freq=30.0):
    """Filter acceleration data using Butterworth filter."""
    sample_rate = 1.0 / dt
    print(f"Filtering acceleration data (cutoff: {cutoff_freq:.2f} Hz, sample rate: {sample_rate:.2f} Hz)")
    ddq_filt = butterworth_filter(ddq, cutoff_freq, sample_rate, order=4)
    return ddq_filt
