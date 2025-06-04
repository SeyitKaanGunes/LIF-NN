import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

# ----------------- Parameters -----------------
T = 100
dt = 1.0
time_steps = int(T / dt)

input_neurons    = 784
hidden_neurons   = 200
output_neurons   = 10
max_current      = 8.0       # Increased from 5.0 to generate more spikes
initial_threshold= -65.0     # Lowered to make neurons more sensitive initially
homeo_lr_thr     = 0.0001    # Significantly reduced for more stability
target_spikes    = 10        # Increased from 5 to encourage more activity
num_epochs       = 15
num_samples      = 5000
test_samples     = 200
stdp_lr          = 0.01      # Increased from 0.004 for stronger learning
stdp_window      = 20
stdp_window_tau  = 10.0
lateral_strength = 5.0       # Reduced from 7.0 for less inhibition

# Refined output layer learning parameters
teacher_forcing_strength = 0.1    # Significantly increased from 0.03
ltd_factor               = 0.5    # Reduced from 0.8 to further limit depression
target_spikes_o_correct_factor = 1.5  # Increased to encourage more activity for correct class
target_spikes_o_incorrect_factor = 0.3  # Increased slightly from 0.2

# Weight stability parameters
weight_stability_factor = 0.005  # Reduced from 0.01 for less stability constraint
min_weight_value = 0.01  # Increased from 0.001 to ensure stronger connections

# ----------------- Load MNIST -----------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

# ----------------- Weights & Thresholds -----------------
np.random.seed(19)  # For consistency

# Input to Hidden Weights - Stronger initialization
W_ih = np.random.normal(0.2, 0.2, (hidden_neurons, input_neurons))  # Mean shifted to positive
for i in range(hidden_neurons):
    norm = np.linalg.norm(W_ih[i])
    if norm > 1e-6:
        W_ih[i] /= norm
    W_ih[i] *= 0.7  # Increased from 0.5 for stronger initial connections

# Hidden to Output Weights - Stronger initialization
W_ho = np.random.uniform(0.05, 0.3, (output_neurons, hidden_neurons))  # Increased range

# Initialize with class-specific biases to break symmetry
for i in range(output_neurons):
    # Find examples of this digit
    digit_samples = np.where(y_train == i)[0][:5]
    if len(digit_samples) > 0:
        for sample_idx in digit_samples:
            image = x_train[sample_idx].flatten()
            # Create hidden layer activation based on this sample
            h_activation = np.tanh(W_ih @ image * 3.0)  # Stronger activation
            # Bias the output weights towards this activation
            W_ho[i] += 0.05 * (h_activation > 0)  # Binary bias based on activation sign

# Normalize W_ho initially
for i in range(output_neurons):
    norm = np.linalg.norm(W_ho[i])
    if norm > 1e-6:
        W_ho[i] /= norm
    W_ho[i] *= 0.5  # Increased from 0.3 for stronger initial connections

# Initialize thresholds with class-specific variations to break symmetry
V_thresholds_h = np.random.uniform(initial_threshold-2.0, initial_threshold+2.0, hidden_neurons)
V_thresholds_o = np.random.uniform(initial_threshold-2.0, initial_threshold+2.0, output_neurons)

# For tracking training stability
accuracy_history = []
weight_norms_history = []
threshold_history = []
spike_counts_history = []

# ----------------- Spike Encoding Function -----------------
def temporal_encode(image, steps, noise_level=0.05):
    """Convert image to spike train using temporal coding with jitter"""
    # Invert pixel values (darker pixels fire earlier)
    inverted = 1.0 - image

    # Create base firing times (darker pixels fire earlier)
    fire_times = np.round(inverted * steps * 0.8).astype(int)  # Compress to first 80% of time window

    # Add small random jitter for robustness
    jitter = np.random.randint(-2, 3, size=fire_times.shape)
    fire_times = np.clip(fire_times + jitter, 0, steps-1)

    # Create spike train
    spike_train = np.zeros((steps, len(image)))

    # Set spikes at appropriate times
    for i_enc in range(len(image)):
        if image[i_enc] > 0.05:  # Only create spikes for non-background pixels
            t_fire = fire_times[i_enc]
            spike_train[t_fire, i_enc] = 1.0

            # Add a second spike for stronger pixels
            if image[i_enc] > 0.5 and t_fire + 10 < steps:
                spike_train[t_fire + 10, i_enc] = 1.0

            # Add noise spikes with low probability
            if np.random.rand() < noise_level:
                noise_t = np.random.randint(0, steps)
                spike_train[noise_t, i_enc] = 1.0

    return spike_train

# ----------------- Training Loop -----------------
print("Starting training...")
# Debug class to monitor
debug_label_non_zero = 1
debug_label_zero = 0

# Pre-training evaluation to check initial state
initial_correct = 0
for idx_init in range(min(100, test_samples)):
    image_init = x_test[idx_init].flatten()
    true_label_init = y_test[idx_init]
    normalized_init = np.clip(image_init * 2.0, 0, 1.0)
    spike_input_init = temporal_encode(normalized_init, time_steps, noise_level=0.0)

    V_h_init = np.full((time_steps, hidden_neurons), -70.0)
    V_o_init = np.full((time_steps, output_neurons), -70.0)
    spikes_h_init = np.zeros_like(V_h_init, dtype=bool)
    spikes_o_init = np.zeros_like(V_o_init, dtype=bool)

    for t_init in range(1, time_steps):
        I_syn_h_init = W_ih @ spike_input_init[t_init-1] * max_current
        dV_h_init = (-(V_h_init[t_init-1] + 70.0) + I_syn_h_init) / 10.0
        V_h_init[t_init] = V_h_init[t_init-1] + dt * dV_h_init
        spikes_h_init[t_init] = V_h_init[t_init] >= V_thresholds_h
        V_h_init[t_init, spikes_h_init[t_init]] = -80.0

        I_syn_o_init = W_ho @ spikes_h_init[t_init].astype(float) * max_current
        dV_o_init = (-(V_o_init[t_init-1] + 70.0) + I_syn_o_init) / 10.0
        V_o_init[t_init] = V_o_init[t_init-1] + dt * dV_o_init
        spikes_o_init[t_init] = V_o_init[t_init] >= V_thresholds_o
        V_o_init[t_init, spikes_o_init[t_init]] = -80.0

    counts_init = spikes_o_init.sum(axis=0)
    if np.sum(counts_init) > 0:
        predicted_label_init = np.argmax(counts_init)
    else:
        predicted_label_init = np.argmax(V_o_init[-1])

    if predicted_label_init == true_label_init:
        initial_correct += 1

initial_accuracy = 100 * initial_correct / min(100, test_samples)
print(f"Initial accuracy (before training): {initial_accuracy:.2f}%")
print(f"Initial hidden layer thresholds: min={np.min(V_thresholds_h):.2f}, max={np.max(V_thresholds_h):.2f}, mean={np.mean(V_thresholds_h):.2f}")
print(f"Initial output layer thresholds: min={np.min(V_thresholds_o):.2f}, max={np.max(V_thresholds_o):.2f}, mean={np.mean(V_thresholds_o):.2f}")

for epoch in range(1, num_epochs+1):
    epoch_spikes_h_total = np.zeros(hidden_neurons)
    epoch_spikes_o_total = np.zeros(output_neurons)
    epoch_correct_predictions = 0
    epoch_processed_samples = 0

    # Store initial weights for stability tracking
    W_ho_start_epoch = W_ho.copy()
    W_ih_start_epoch = W_ih.copy()

    # Store initial thresholds for tracking
    V_thresholds_h_start = V_thresholds_h.copy()
    V_thresholds_o_start = V_thresholds_o.copy()

    indices = np.random.permutation(len(x_train))
    # Gentler learning rate decay
    current_stdp_lr = stdp_lr * (1.0 - 0.2 * epoch / num_epochs)  # Reduced from 0.3 to 0.2
    current_homeo_lr = homeo_lr_thr * (1.0 - 0.2 * epoch / num_epochs)  # Also gentler decay

    # Batch processing for weight normalization
    batch_size = 200  # Increased from 100 to 200

    # For monitoring spike activity distribution
    all_sample_spikes_h = []
    all_sample_spikes_o = []

    for sample_idx_loop in range(num_samples):
        idx = indices[sample_idx_loop]
        image = x_train[idx].flatten()
        label = y_train[idx]

        normalized = np.clip(image * 2.0, 0, 1.0)  # Increased contrast from 1.5 to 2.0
        spike_input = temporal_encode(normalized, time_steps)

        if np.sum(spike_input) == 0:  # Skip if no input spikes (empty image, etc.)
            continue

        V_h = np.full((time_steps, hidden_neurons), -70.0)
        V_o = np.full((time_steps, output_neurons), -70.0)
        spikes_h = np.zeros_like(V_h, dtype=bool)
        spikes_o = np.zeros_like(V_o, dtype=bool)
        tau_m = 10.0

        for t in range(1, time_steps):
            # Hidden layer dynamics
            I_syn_h = W_ih @ spike_input[t-1] * max_current
            dV_h = (-(V_h[t-1] + 70.0) + I_syn_h) / tau_m
            V_h[t] = V_h[t-1] + dt * dV_h
            spikes_h[t] = V_h[t] >= V_thresholds_h
            V_h[t, spikes_h[t]] = -80.0

            # Output layer dynamics
            I_syn_o = W_ho @ spikes_h[t].astype(float) * max_current
            dV_o = (-(V_o[t-1] + 70.0) + I_syn_o) / tau_m
            V_o[t] = V_o[t-1] + dt * dV_o
            spikes_o[t] = V_o[t] >= V_thresholds_o
            V_o[t, spikes_o[t]] = -80.0

            # Softer lateral inhibition
            if np.any(spikes_o[t]):
                active_indices = np.where(spikes_o[t])[0]
                if len(active_indices) > 1:
                    for i_li in range(output_neurons):
                        if i_li not in active_indices:  # Non-firing neurons
                            V_o[t, i_li] -= lateral_strength * 0.5  # Less inhibition

            # STDP for hidden layer
            if t > 1 and np.any(spikes_h[t]):
                for h_idx in np.where(spikes_h[t])[0]:
                    for t_pre_input in range(max(0, t - stdp_window), t):
                        for i_idx in np.where(spike_input[t_pre_input] > 0)[0]:
                            time_diff = (t - t_pre_input) * dt
                            # Weight-dependent LTP with stability factor
                            delta_w = current_stdp_lr * np.exp(-time_diff / stdp_window_tau) * (1.0 - W_ih[h_idx, i_idx])

                            # Add weight stability in later epochs
                            if epoch > num_epochs // 2:
                                stability_factor = weight_stability_factor * (epoch - num_epochs // 2) / (num_epochs // 2)
                                delta_w *= (1.0 - stability_factor)

                            W_ih[h_idx, i_idx] += delta_w

            # STDP and supervised learning for output layer
            if t > 1:
                for o_idx in range(output_neurons):
                    # Teacher forcing for correct class
                    if o_idx == label:
                        if not spikes_o[t, o_idx] and np.sum(spikes_h[t]) > 0:  # Correct neuron didn't fire AND hidden layer activity exists
                            for h_idx_tf in np.where(spikes_h[t])[0]:  # Currently active hidden neurons
                                delta_w = current_stdp_lr * teacher_forcing_strength * (1.0 - W_ho[o_idx, h_idx_tf])

                                # Add weight stability in later epochs
                                if epoch > num_epochs // 2:
                                    stability_factor = weight_stability_factor * (epoch - num_epochs // 2) / (num_epochs // 2)
                                    delta_w *= (1.0 - stability_factor)

                                W_ho[o_idx, h_idx_tf] += delta_w

                    # Standard STDP for output neurons that fire
                    if spikes_o[t, o_idx]:
                        for t_pre_h in range(max(0, t - stdp_window), t):
                            for h_idx_stdp in np.where(spikes_h[t_pre_h])[0]:
                                time_diff = (t - t_pre_h) * dt
                                if o_idx == label:  # LTP for correct class
                                    delta_w = current_stdp_lr * np.exp(-time_diff / stdp_window_tau) * (1.0 - W_ho[o_idx, h_idx_stdp])

                                    # Add weight stability in later epochs
                                    if epoch > num_epochs // 2:
                                        stability_factor = weight_stability_factor * (epoch - num_epochs // 2) / (num_epochs // 2)
                                        delta_w *= (1.0 - stability_factor)

                                    W_ho[o_idx, h_idx_stdp] += delta_w
                                else:  # LTD for incorrect classes
                                    delta_w = current_stdp_lr * ltd_factor * np.exp(-time_diff / stdp_window_tau) * W_ho[o_idx, h_idx_stdp]

                                    # Add weight stability in later epochs
                                    if epoch > num_epochs // 2:
                                        stability_factor = weight_stability_factor * (epoch - num_epochs // 2) / (num_epochs // 2)
                                        delta_w *= (1.0 - stability_factor)

                                    W_ho[o_idx, h_idx_stdp] -= delta_w

        # Weight constraints
        W_ih = np.clip(W_ih, 0, 1.0)
        W_ho = np.clip(W_ho, min_weight_value, 1.0)  # Prevent weights from becoming too small

        # Normalize weights less frequently (every batch_size samples)
        if sample_idx_loop % batch_size == batch_size - 1 or sample_idx_loop == num_samples - 1:
            # L2 normalization for hidden layer
            for h_norm_idx in range(hidden_neurons):
                norm = np.linalg.norm(W_ih[h_norm_idx])
                if norm > 1e-6:
                    W_ih[h_norm_idx] = (W_ih[h_norm_idx] / norm) * 0.7  # Set norm to 0.7

            # L2 normalization for output layer
            for o_norm_idx in range(output_neurons):
                norm = np.linalg.norm(W_ho[o_norm_idx])
                if norm > 1e-6:
                    W_ho[o_norm_idx] = (W_ho[o_norm_idx] / norm) * 0.5  # Set norm to 0.5

        # Count spikes and update statistics
        sample_spikes_h_sum = spikes_h.sum(axis=0)
        sample_spikes_o_sum = spikes_o.sum(axis=0)

        # Store spike counts for distribution analysis
        all_sample_spikes_h.append(np.sum(sample_spikes_h_sum))
        all_sample_spikes_o.append(sample_spikes_o_sum)

        epoch_spikes_h_total += sample_spikes_h_sum
        epoch_spikes_o_total += sample_spikes_o_sum

        # Prediction (for monitoring during training)
        if np.sum(sample_spikes_o_sum) > 0:
            predicted_label_train = np.argmax(sample_spikes_o_sum)
        else:
            predicted_label_train = np.argmax(V_o[-1])  # Based on final potential

        if predicted_label_train == label:
            epoch_correct_predictions += 1
        epoch_processed_samples += 1

        # Homeostatic threshold adjustment for hidden layer
        V_thresholds_h += current_homeo_lr * (sample_spikes_h_sum - target_spikes)
        V_thresholds_h = np.clip(V_thresholds_h, -70.0, -45.0)

        # Homeostatic threshold adjustment for output layer with class-specific targets
        for o_homeo_idx in range(output_neurons):
            if o_homeo_idx == label:
                error_o = sample_spikes_o_sum[o_homeo_idx] - target_spikes * target_spikes_o_correct_factor
            else:
                error_o = sample_spikes_o_sum[o_homeo_idx] - target_spikes * target_spikes_o_incorrect_factor

            # Add threshold stability in later epochs
            if epoch > num_epochs // 2:
                stability_factor = weight_stability_factor * (epoch - num_epochs // 2) / (num_epochs // 2)
                error_o *= (1.0 - stability_factor)

            V_thresholds_o[o_homeo_idx] += current_homeo_lr * error_o

        V_thresholds_o = np.clip(V_thresholds_o, -70.0, -45.0)

        # Print progress more frequently
        if sample_idx_loop > 0 and sample_idx_loop % 200 == 0:
            current_train_acc = epoch_correct_predictions / epoch_processed_samples if epoch_processed_samples > 0 else 0
            print(f"  Epoch {epoch}, Sample {sample_idx_loop}/{num_samples}, Train Acc (batch): {current_train_acc*100:.2f}%")
            print(f"    Output spikes: {np.round(epoch_spikes_o_total/max(1, epoch_processed_samples), 2)}")
            print(f"    Output thresholds: {np.round(V_thresholds_o, 2)}")

    # Calculate weight change magnitude for stability tracking
    w_ho_change = np.mean(np.abs(W_ho - W_ho_start_epoch))
    w_ih_change = np.mean(np.abs(W_ih - W_ih_start_epoch))

    # Calculate threshold change magnitude
    v_th_h_change = np.mean(np.abs(V_thresholds_h - V_thresholds_h_start))
    v_th_o_change = np.mean(np.abs(V_thresholds_o - V_thresholds_o_start))

    # Calculate and store training accuracy for this epoch
    train_accuracy_epoch = epoch_correct_predictions / epoch_processed_samples if epoch_processed_samples > 0 else 0
    accuracy_history.append(train_accuracy_epoch * 100)

    # Store weight norms for tracking
    weight_norms_history.append({
        'hidden': np.mean([np.linalg.norm(W_ih[i]) for i in range(hidden_neurons)]),
        'output': np.mean([np.linalg.norm(W_ho[i]) for i in range(output_neurons)]),
        'w_ho_change': w_ho_change,
        'w_ih_change': w_ih_change
    })

    # Store threshold values
    threshold_history.append({
        'hidden_mean': np.mean(V_thresholds_h),
        'output_mean': np.mean(V_thresholds_o),
        'output_min': np.min(V_thresholds_o),
        'output_max': np.max(V_thresholds_o),
        'v_th_h_change': v_th_h_change,
        'v_th_o_change': v_th_o_change
    })

    # Store spike counts
    spike_counts_history.append({
        'hidden_mean': np.mean(epoch_spikes_h_total) / epoch_processed_samples if epoch_processed_samples > 0 else 0,
        'output_total': epoch_spikes_o_total / epoch_processed_samples if epoch_processed_samples > 0 else np.zeros(output_neurons),
        'hidden_distribution': all_sample_spikes_h
    })

    print(f"Epoch {epoch} complete. Training Accuracy: {train_accuracy_epoch*100:.2f}%")
    print(f"  Avg Hidden layer spike counts (per neuron for epoch): {epoch_spikes_h_total.mean()/epoch_processed_samples:.2f}")
    print(f"  Avg Output layer spike counts (per neuron for epoch): {epoch_spikes_o_total/epoch_processed_samples}")
    print(f"  Final Output thresholds for epoch: {np.round(V_thresholds_o, 2)}")
    print(f"  Weight change magnitude - Hidden: {w_ih_change:.4f}, Output: {w_ho_change:.4f}")
    print(f"  Threshold change magnitude - Hidden: {v_th_h_change:.4f}, Output: {v_th_o_change:.4f}")

    # Mid-training evaluation (every 3 epochs)
    if epoch % 3 == 0:
        mid_correct = 0
        for idx_mid in range(min(50, test_samples)):
            image_mid = x_test[idx_mid].flatten()
            true_label_mid = y_test[idx_mid]
            normalized_mid = np.clip(image_mid * 2.0, 0, 1.0)
            spike_input_mid = temporal_encode(normalized_mid, time_steps, noise_level=0.0)

            V_h_mid = np.full((time_steps, hidden_neurons), -70.0)
            V_o_mid = np.full((time_steps, output_neurons), -70.0)
            spikes_h_mid = np.zeros_like(V_h_mid, dtype=bool)
            spikes_o_mid = np.zeros_like(V_o_mid, dtype=bool)

            for t_mid in range(1, time_steps):
                I_syn_h_mid = W_ih @ spike_input_mid[t_mid-1] * max_current
                dV_h_mid = (-(V_h_mid[t_mid-1] + 70.0) + I_syn_h_mid) / 10.0
                V_h_mid[t_mid] = V_h_mid[t_mid-1] + dt * dV_h_mid
                spikes_h_mid[t_mid] = V_h_mid[t_mid] >= V_thresholds_h
                V_h_mid[t_mid, spikes_h_mid[t_mid]] = -80.0

                I_syn_o_mid = W_ho @ spikes_h_mid[t_mid].astype(float) * max_current
                dV_o_mid = (-(V_o_mid[t_mid-1] + 70.0) + I_syn_o_mid) / 10.0
                V_o_mid[t_mid] = V_o_mid[t_mid-1] + dt * dV_o_mid
                spikes_o_mid[t_mid] = V_o_mid[t_mid] >= V_thresholds_o
                V_o_mid[t_mid, spikes_o_mid[t_mid]] = -80.0

            counts_mid = spikes_o_mid.sum(axis=0)
            if np.sum(counts_mid) > 0:
                predicted_label_mid = np.argmax(counts_mid)
            else:
                predicted_label_mid = np.argmax(V_o_mid[-1])

            if predicted_label_mid == true_label_mid:
                mid_correct += 1

        mid_accuracy = 100 * mid_correct / min(50, test_samples)
        print(f"  Mid-training accuracy (epoch {epoch}): {mid_accuracy:.2f}%")

# ----------------- Testing and Accuracy -----------------
print(f"\nTesting on {test_samples} samples...")
correct = 0
confusion_matrix = np.zeros((10, 10), dtype=int)

for idx_test in range(test_samples):
    image_test = x_test[idx_test].flatten()
    true_label_test = y_test[idx_test]
    normalized_test = np.clip(image_test * 2.0, 0, 1.0)
    spike_input_test = temporal_encode(normalized_test, time_steps, noise_level=0.0)  # No noise in testing

    V_h_test = np.full((time_steps, hidden_neurons), -70.0)
    V_o_test = np.full((time_steps, output_neurons), -70.0)
    spikes_h_test = np.zeros_like(V_h_test, dtype=bool)
    spikes_o_test = np.zeros_like(V_o_test, dtype=bool)
    tau_m_test = 10.0

    for t_test in range(1, time_steps):
        I_syn_h_test = W_ih @ spike_input_test[t_test-1] * max_current
        dV_h_test = (-(V_h_test[t_test-1] + 70.0) + I_syn_h_test) / tau_m_test
        V_h_test[t_test] = V_h_test[t_test-1] + dt * dV_h_test
        spikes_h_test[t_test] = V_h_test[t_test] >= V_thresholds_h  # Trained thresholds
        V_h_test[t_test, spikes_h_test[t_test]] = -80.0

        I_syn_o_test = W_ho @ spikes_h_test[t_test].astype(float) * max_current
        dV_o_test = (-(V_o_test[t_test-1] + 70.0) + I_syn_o_test) / tau_m_test
        V_o_test[t_test] = V_o_test[t_test-1] + dt * dV_o_test
        spikes_o_test[t_test] = V_o_test[t_test] >= V_thresholds_o  # Trained thresholds
        V_o_test[t_test, spikes_o_test[t_test]] = -80.0

        # Apply lateral inhibition in testing too
        if np.any(spikes_o_test[t_test]):
            active_indices_test = np.where(spikes_o_test[t_test])[0]
            if len(active_indices_test) > 1:
                potentials_at_spike = V_o_test[t_test, active_indices_test]
                winner_index_in_active = np.argmax(potentials_at_spike)
                winner_neuron = active_indices_test[winner_index_in_active]
                for i_li_test in range(output_neurons):
                    if i_li_test != winner_neuron:  # Inhibit all except winner
                        V_o_test[t_test, i_li_test] -= lateral_strength

    counts_test = spikes_o_test.sum(axis=0)
    if np.sum(counts_test) > 0:
        # More robust: Use highest spike count AND highest final potential
        max_spike_count = np.max(counts_test)
        candidates = np.where(counts_test == max_spike_count)[0]
        if len(candidates) == 1:
            predicted_label = candidates[0]
        else:
            predicted_label = candidates[np.argmax(V_o_test[-1, candidates])]
    else:
        predicted_label = np.argmax(V_o_test[-1])

    confusion_matrix[true_label_test, predicted_label] += 1
    if predicted_label == true_label_test:
        correct += 1

    if idx_test % 20 == 0:
        print(f"  Testing sample {idx_test}/{test_samples}")

accuracy = 100 * correct / test_samples if test_samples > 0 else 0.0

# ----------------- Visualizations -----------------
# Training accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), accuracy_history, 'o-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy (%)')
plt.title('Training Accuracy Over Epochs')
plt.grid(True)
plt.savefig('training_accuracy.png')

# Weight norms over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [d['hidden'] for d in weight_norms_history], 'o-', label='Hidden Layer')
plt.plot(range(1, num_epochs+1), [d['output'] for d in weight_norms_history], 's-', label='Output Layer')
plt.xlabel('Epoch')
plt.ylabel('Average Weight Norm')
plt.title('Weight Norms Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('weight_norms.png')

# Weight change magnitude over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [d['w_ih_change'] for d in weight_norms_history], 'o-', label='Hidden Layer')
plt.plot(range(1, num_epochs+1), [d['w_ho_change'] for d in weight_norms_history], 's-', label='Output Layer')
plt.xlabel('Epoch')
plt.ylabel('Weight Change Magnitude')
plt.title('Weight Stability Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('weight_stability.png')

# Threshold values over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [d['hidden_mean'] for d in threshold_history], 'o-', label='Hidden Layer (Mean)')
plt.plot(range(1, num_epochs+1), [d['output_mean'] for d in threshold_history], 's-', label='Output Layer (Mean)')
plt.plot(range(1, num_epochs+1), [d['output_min'] for d in threshold_history], '^--', label='Output Layer (Min)')
plt.plot(range(1, num_epochs+1), [d['output_max'] for d in threshold_history], 'v--', label='Output Layer (Max)')
plt.xlabel('Epoch')
plt.ylabel('Threshold Value')
plt.title('Neuron Thresholds Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('thresholds.png')

# Threshold change magnitude over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [d['v_th_h_change'] for d in threshold_history], 'o-', label='Hidden Layer')
plt.plot(range(1, num_epochs+1), [d['v_th_o_change'] for d in threshold_history], 's-', label='Output Layer')
plt.xlabel('Epoch')
plt.ylabel('Threshold Change Magnitude')
plt.title('Threshold Stability Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('threshold_stability.png')

# Spike counts over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), [d['hidden_mean'] for d in spike_counts_history], 'o-', label='Hidden Layer (Mean)')
for i in range(output_neurons):
    plt.plot(range(1, num_epochs+1), [d['output_total'][i] for d in spike_counts_history], '--', label=f'Output {i}')
plt.xlabel('Epoch')
plt.ylabel('Average Spike Count')
plt.title('Neuron Activity Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('spike_activity.png')

# Hidden layer weights visualization
plt.figure(figsize=(10, 8))
for i in range(min(16, hidden_neurons)):  # Show first 16 hidden neurons
    plt.subplot(4, 4, i+1)
    weight_img = W_ih[i].reshape(28, 28)
    plt.imshow(weight_img, cmap='viridis')
    plt.title(f"Hidden {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('hidden_weights.png')

# Output layer weights visualization
plt.figure(figsize=(12, 5))
for i in range(output_neurons):
    plt.subplot(2, 5, i+1)
    plt.bar(range(min(50, hidden_neurons)), W_ho[i, :min(50, hidden_neurons)])
    plt.title(f"Output {i}")
    plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('output_weights.png')

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Weight histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(W_ih.flatten(), bins=50, alpha=0.7)
plt.title('Hidden Layer Weights')
plt.xlabel('Weight Value')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.hist(W_ho.flatten(), bins=50, alpha=0.7)
plt.title('Output Layer Weights')
plt.xlabel('Weight Value')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('weight_histograms.png')

print(f"\n📊 Test Accuracy ({test_samples} samples): {accuracy:.2f}%")
print("Visualizations saved as PNG files.")

# Save the trained model weights and thresholds
np.save('snn_weights_ih.npy', W_ih)
np.save('snn_weights_ho.npy', W_ho)
np.save('snn_thresholds_h.npy', V_thresholds_h)
np.save('snn_thresholds_o.npy', V_thresholds_o)
print("Model weights and thresholds saved.")
