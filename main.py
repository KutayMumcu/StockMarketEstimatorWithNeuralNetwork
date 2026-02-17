import numpy as np
import pandas as pd
import sys


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


# ==========================================
# 2. DATA LOADING AND INDICATORS
# ==========================================
print("--- DYNAMIC RETURN PREDICTION SYSTEM (SMART RETURN PREDICTION) ---")
FILE_PATH = 'data.xlsx'  # Enter file name here

try:
    df_full = pd.read_excel(FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: '{FILE_PATH}' not found.")
    sys.exit()

# Select necessary columns
df = df_full[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']].copy()

# Feature Engineering (Before slicing data)
print("Calculating indicators...")
df['VOL'] = np.log1p(df['VOL'])  # Logarithmic Volume
df['RSI'] = calculate_rsi(df['CLOSE'])
df['MA_5'] = df['CLOSE'].rolling(window=5).mean() / df['CLOSE']
df['MA_20'] = df['CLOSE'].rolling(window=20).mean() / df['CLOSE']

df.dropna(inplace=True)  # Clean NaN values

# ==========================================
# 3. USER INTERFACE (DYNAMIC INPUT)
# ==========================================
print("-" * 40)
print(f"Current Data Count: {len(df)} rows.")
print("SELECT MODE:")
print("1 - TEST MODE (Backtest - Test Past Data)")
print("2 - REAL MODE (Future Prediction)")

try:
    mode_selection = int(input("Your Choice (1 or 2): "))
    WINDOW_SIZE = int(input("Window Size (How many past days? e.g., 20): "))
    PREDICTION_HORIZON = int(input("Prediction Horizon (How many days ahead? e.g., 10): "))
except ValueError:
    print("Please enter numbers only.")
    sys.exit()

# Data Slicing
working_df = df.copy()
ground_truth_available = False
reference_date = ""
actual_price = 0

if mode_selection == 1:
    try:
        limit_rows = int(input(f"How many rows of data to use? (Max {len(df)}): "))
        working_df = df.iloc[:limit_rows].copy()

        # Store ground truth for testing
        target_idx = limit_rows + PREDICTION_HORIZON - 1
        if target_idx < len(df):
            actual_price = df.iloc[target_idx]['CLOSE']
            reference_date = df.iloc[target_idx]['DATE']
            ground_truth_available = True
        else:
            print("WARNING: Selected range exceeds file end, validation cannot be performed.")
    except ValueError:
        sys.exit()

# ==========================================
# 4. DATA PREPARATION (PROPORTIONAL CHANGE MATH)
# ==========================================
# 8 Features: OPEN, HIGH, LOW, CLOSE, MA_5, MA_20, VOL, RSI
features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MA_5', 'MA_20', 'VOL', 'RSI']
data_values = working_df[features].values

X_list = []
y_list = []

limit = len(data_values) - WINDOW_SIZE - PREDICTION_HORIZON + 1
if limit < 1:
    print("ERROR: Data size is insufficient for the selected window size and prediction horizon!")
    sys.exit()

for i in range(limit):
    # Get Window
    window_raw = data_values[i: i + WINDOW_SIZE].copy()

    # Reference: Last close price of the window
    reference_price = window_raw[-1, 3]

    # 1. Normalization: (Value / Reference) - 1
    for col_idx in [0, 1, 2, 3, 4, 5]:
        window_raw[:, col_idx] = (window_raw[:, col_idx] / reference_price) - 1.0

    # Simple normalization for Vol and RSI
    window_raw[:, 6] = window_raw[:, 6] / (np.max(window_raw[:, 6]) + 1e-9)
    window_raw[:, 7] = window_raw[:, 7] / 100.0

    X_list.append(window_raw.flatten())

    # TARGET: Percentage Change
    future_price = data_values[i + WINDOW_SIZE + PREDICTION_HORIZON - 1, 3]
    change_rate = (future_price - reference_price) / reference_price
    y_list.append(change_rate)

X = np.array(X_list)
y = np.array(y_list).reshape(-1, 1)

# Shuffle
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_train = X[indices]
y_train = y[indices]

# ==========================================
# 5. DYNAMIC ARCHITECTURE (AUTO-SCALING)
# ==========================================
input_dim = X.shape[1]  # Input dimension (Window * 8)
sample_size = X.shape[0]  # Number of samples

# SMART NEURON CALCULATION:
# If data is scarce (e.g., 750 rows), reduce neuron count (Prevent Overfitting).
# If data is abundant, increase neuron count.
if sample_size < 1000:
    H1 = min(64, int(input_dim * 0.8))  # Small dataset setting
    H2 = H1 // 2
    LR = 1e-3  # Learn faster if data is scarce
    EPOCH = 3000
elif sample_size < 5000:
    H1 = 128
    H2 = 64
    LR = 1e-4  # Medium dataset setting
    EPOCH = 4000
else:
    H1 = 256  # Large dataset setting
    H2 = 128
    LR = 5e-5  # Precise learning
    EPOCH = 5000

# Minimum safety limits
H1 = max(32, H1)
H2 = max(16, H2)

print(f"\n--- Model Architecture (Data Count: {sample_size}) ---")
print(f"Input: {input_dim} -> H1: {H1} -> H2: {H2} -> Output: 1")

# ==========================================
# 6. TRAINING
# ==========================================
np.random.seed(42)
D_in = input_dim
D_out = 1

w1 = np.random.randn(D_in, H1) * np.sqrt(2. / D_in)
b1 = np.zeros((1, H1))
w2 = np.random.randn(H1, H2) * np.sqrt(2. / H1)
b2 = np.zeros((1, H2))
w3 = np.random.randn(H2, D_out) * np.sqrt(2. / H2)
b3 = np.zeros((1, D_out))

print("Training started...")
for t in range(EPOCH):
    # Forward
    z1 = X_train.dot(w1) + b1
    h1 = np.maximum(0, z1)
    z2 = h1.dot(w2) + b2
    h2 = np.maximum(0, z2)
    y_pred = h2.dot(w3) + b3

    loss = np.square(y_pred - y_train).mean()

    # Backward
    grad_y_pred = 2.0 * (y_pred - y_train) / len(X_train)

    grad_w3 = h2.T.dot(grad_y_pred)
    grad_b3 = np.sum(grad_y_pred, axis=0, keepdims=True)

    grad_h2 = grad_y_pred.dot(w3.T)
    grad_z2 = grad_h2.copy()
    grad_z2[z2 < 0] = 0
    grad_w2 = h1.T.dot(grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

    grad_h1 = grad_z2.dot(w2.T)
    grad_z1 = grad_h1.copy()
    grad_z1[z1 < 0] = 0
    grad_w1 = X_train.T.dot(grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

    w1 -= LR * grad_w1
    b1 -= LR * grad_b1
    w2 -= LR * grad_w2
    b2 -= LR * grad_b2
    w3 -= LR * grad_w3
    b3 -= LR * grad_b3

    if t % 1000 == 0:
        print(f"Epoch {t} | Loss: {loss:.6f}")

# ==========================================
# 7. PREDICTION AND REPORTING
# ==========================================
print("\n" + "=" * 40)
print("   RESULT REPORT")
print("=" * 40)

# Prepare Last Window
last_window_raw = data_values[-WINDOW_SIZE:].copy()
reference_price_last = last_window_raw[-1, 3]

# Normalization
for col_idx in [0, 1, 2, 3, 4, 5]:
    last_window_raw[:, col_idx] = (last_window_raw[:, col_idx] / reference_price_last) - 1.0

last_window_raw[:, 6] = last_window_raw[:, 6] / (np.max(last_window_raw[:, 6]) + 1e-9)
last_window_raw[:, 7] = last_window_raw[:, 7] / 100.0

input_vector = last_window_raw.flatten().reshape(1, -1)

# Prediction
h1_val = np.maximum(0, input_vector.dot(w1) + b1)
h2_val = np.maximum(0, h1_val.dot(w2) + b2)
tahmin_oran = h2_val.dot(w3) + b3
predicted_change = tahmin_oran[0][0]

# Convert to Price
predicted_price = reference_price_last * (1 + predicted_change)

print(f"Reference Price (Last Day) : {reference_price_last:.2f}")
print(f"Expected Change            : %{predicted_change * 100:.2f}")
print(f"MODEL PREDICTION           : {predicted_price:.2f}")

if mode_selection == 1 and ground_truth_available:
    print("-" * 30)
    print(f"Actual Date                : {reference_date}")
    print(f"Actual Price               : {actual_price:.2f}")

    error_percentage = (abs(predicted_price - actual_price) / actual_price) * 100

    # Direction Check
    model_direction = "UP" if predicted_price > reference_price_last else "DOWN"
    actual_direction = "UP" if actual_price > reference_price_last else "DOWN"

    direction_status = "✅ DIRECTION CORRECT" if model_direction == actual_direction else "❌ DIRECTION WRONG"
    success_status = "✅ SUCCESSFUL" if error_percentage < 5 else "⚠️ DEVIATION DETECTED"

    print(f"Error Rate                 : %{error_percentage:.2f}")
    print(f"General Status             : {success_status}")
    print(f"Direction Prediction       : {direction_status}")

elif mode_selection == 2:
    direction_icon = "🔼 UP" if predicted_change > 0 else "🔽 DOWN"
    print(f"Direction Expectation      : {direction_icon}")