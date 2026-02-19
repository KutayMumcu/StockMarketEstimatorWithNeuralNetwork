import os
import glob
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
# 2. USER INTERFACE (DYNAMIC INPUT)
# ==========================================
print("--- DYNAMIC RETURN PREDICTION SYSTEM (SMART RETURN PREDICTION) ---")
print("-" * 40)
print("SELECT MODE:")
print("1 - TEST MODE (Backtest - Test Past Data / Single File)")
print("2 - REAL MODE (Bulk Scan or Single File - Future Prediction)")

try:
    mode_selection = int(input("Your Choice (1 or 2): "))
    WINDOW_SIZE = int(input("Window Size (How many past days? e.g., 20): "))
    PREDICTION_HORIZON = int(input("Prediction Horizon (How many days ahead? e.g., 10): "))
except ValueError:
    print("Please enter numbers only.")
    sys.exit()

files_to_process = []
limit_rows = None
threshold = None

if mode_selection == 1:
    file_path = input("Enter the full name of the file to be tested (e.g., THYAO.xlsx): ")
    files_to_process.append(file_path)
    try:
        limit_input = input("How many rows of data to use? (You can type 0 or press Enter for the whole file): ")
        limit_rows = int(limit_input) if limit_input else 0
    except ValueError:
        limit_rows = 0

elif mode_selection == 2:
    path_input = input(
        "Enter the path of the folder OR a single stock file to scan (e.g., C:/stocks or C:/stocks/THYAO.xlsx): ")
    try:
        threshold = float(input(
            "Enter the target minimum percentage increase threshold (You can enter -100 to see all results, e.g., type 5 for 5%): "))
    except ValueError:
        print("You entered an invalid threshold value.")
        sys.exit()

    # If the entered path is a .xlsx file, add only that to the list
    if os.path.isfile(path_input) and path_input.lower().endswith('.xlsx'):
        files_to_process = [path_input]
        print(f"Scanning a single file: {os.path.basename(path_input)}")

    # If the entered path is a folder, find all .xlsx files in it
    elif os.path.isdir(path_input):
        files_to_process = glob.glob(os.path.join(path_input, "*.xlsx"))
        if not files_to_process:
            print("No .xlsx files found in the specified folder.")
            sys.exit()
        print(f"Scanning a total of {len(files_to_process)} stock files...")

    else:
        print("Invalid folder or file path. Please make sure you entered the path correctly.")
        sys.exit()

else:
    print("Invalid mode selection.")
    sys.exit()

# List to keep stocks that pass the threshold
recommended_stocks = []

# ==========================================
# 3. MAIN PROCESSING LOOP
# ==========================================
for file_path in files_to_process:
    file_name = os.path.basename(file_path)
    print(f"\n>> Processing: {file_name} <<")

    try:
        df_full = pd.read_excel(file_path)
    except Exception as e:
        print(f"ERROR: Could not read '{file_name}'. Detail: {e}")
        continue

    # Data sufficiency check
    if len(df_full) < WINDOW_SIZE + PREDICTION_HORIZON:
        print(f"Warning: Data for '{file_name}' is too short. Skipping...")
        continue

    # Feature Engineering
    df = df_full[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']].copy()
    df['VOL'] = np.log1p(df['VOL'])  # Logarithmic Volume
    df['RSI'] = calculate_rsi(df['CLOSE'])
    df['MA_5'] = df['CLOSE'].rolling(window=5).mean() / df['CLOSE']
    df['MA_20'] = df['CLOSE'].rolling(window=20).mean() / df['CLOSE']
    df.dropna(inplace=True)

    working_df = df.copy()
    ground_truth_available = False
    reference_date = ""
    actual_price = 0

    if mode_selection == 1 and limit_rows is not None and limit_rows > 0:
        limit_idx = min(limit_rows, len(df))
        working_df = df.iloc[:limit_idx].copy()

        target_idx = limit_idx + PREDICTION_HORIZON - 1
        if target_idx < len(df):
            actual_price = df.iloc[target_idx]['CLOSE']
            reference_date = df.iloc[target_idx]['DATE']
            ground_truth_available = True
        else:
            print("WARNING: Selected range exceeds file end, validation cannot be performed.")

    # ==========================================
    # 4. DATA PREPARATION (PROPORTIONAL CHANGE MATH)
    # ==========================================
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MA_5', 'MA_20', 'VOL', 'RSI']
    data_values = working_df[features].values

    X_list = []
    y_list = []

    limit = len(data_values) - WINDOW_SIZE - PREDICTION_HORIZON + 1
    if limit < 1:
        print(f"Warning: Cleaned data size for '{file_name}' is insufficient! Skipping...")
        continue

    for i in range(limit):
        window_raw = data_values[i: i + WINDOW_SIZE].copy()
        reference_price = window_raw[-1, 3]

        for col_idx in [0, 1, 2, 3, 4, 5]:
            window_raw[:, col_idx] = (window_raw[:, col_idx] / reference_price) - 1.0

        window_raw[:, 6] = window_raw[:, 6] / (np.max(window_raw[:, 6]) + 1e-9)
        window_raw[:, 7] = window_raw[:, 7] / 100.0

        X_list.append(window_raw.flatten())

        future_price = data_values[i + WINDOW_SIZE + PREDICTION_HORIZON - 1, 3]
        change_rate = (future_price - reference_price) / reference_price
        y_list.append(change_rate)

    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_train = X[indices]
    y_train = y[indices]

    # ==========================================
    # 5. DYNAMIC ARCHITECTURE & HYPERPARAMETERS
    # ==========================================
    input_dim = X.shape[1]
    sample_size = X.shape[0]

    # Restricting neuron count for financial data (To prevent overfitting)
    # L2 Regularization coefficient (Lambda). If too high, model can't learn; if too low, it memorizes.
    LAMBDA_L2 = 1e-3

    if sample_size < 1000:
        H1 = 32
        H2 = 16
        LR = 1e-3
        EPOCH = 3000
    elif sample_size < 5000:
        H1 = 64
        H2 = 32
        LR = 1e-4
        EPOCH = 4000
    else:
        H1 = 64  # We do not exceed 64 even with large datasets
        H2 = 32
        LR = 5e-5
        EPOCH = 5000

    # ==========================================
    # 6. TRAINING (MINI-BATCH GRADIENT DESCENT & L2)
    # ==========================================
    np.random.seed(42)
    w1 = np.random.randn(input_dim, H1) * np.sqrt(2. / input_dim)
    b1 = np.zeros((1, H1))
    w2 = np.random.randn(H1, H2) * np.sqrt(2. / H1)
    b2 = np.zeros((1, H2))
    w3 = np.random.randn(H2, 1) * np.sqrt(2. / H2)
    b3 = np.zeros((1, 1))

    # Mini-Batch size: 32 or 64 is ideal for financial data.
    BATCH_SIZE = 32

    # With mini-batch, updates are made (Data Count / Batch Size) times per epoch.
    # Therefore, reducing the total EPOCH count to around 500 - 1000 is sufficient.
    EPOCH = 800 if sample_size < 5000 else 1000

    print(f"Training started ({sample_size} rows, Mini-Batch: {BATCH_SIZE}, L2 Active)...")

    for t in range(EPOCH):
        # Shuffle training data at the beginning of each epoch (Increases stochastic effect)
        epoch_indices = np.arange(X_train.shape[0])
        np.random.shuffle(epoch_indices)
        X_train_shuffled = X_train[epoch_indices]
        y_train_shuffled = y_train[epoch_indices]

        epoch_loss = 0.0

        # Train by dividing the data into chunks of BATCH_SIZE
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch = X_train_shuffled[i:i + BATCH_SIZE]
            y_batch = y_train_shuffled[i:i + BATCH_SIZE]

            # --- FORWARD PASS ---
            z1 = X_batch.dot(w1) + b1
            h1 = np.maximum(0, z1)
            z2 = h1.dot(w2) + b2
            h2 = np.maximum(0, z2)
            y_pred = h2.dot(w3) + b3

            # --- BACKWARD PASS ---
            grad_y_pred = 2.0 * (y_pred - y_batch) / len(X_batch)

            # Gradient calculations with L2 Penalty
            grad_w3 = h2.T.dot(grad_y_pred) + (LAMBDA_L2 * w3)
            grad_b3 = np.sum(grad_y_pred, axis=0, keepdims=True)

            grad_h2 = grad_y_pred.dot(w3.T)
            grad_z2 = grad_h2.copy()
            grad_z2[z2 < 0] = 0
            grad_w2 = h1.T.dot(grad_z2) + (LAMBDA_L2 * w2)
            grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)

            grad_h1 = grad_z2.dot(w2.T)
            grad_z1 = grad_h1.copy()
            grad_z1[z1 < 0] = 0
            grad_w1 = X_batch.T.dot(grad_z1) + (LAMBDA_L2 * w1)
            grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

            # --- UPDATE WEIGHTS ---
            w1 -= LR * grad_w1
            b1 -= LR * grad_b1
            w2 -= LR * grad_w2
            b2 -= LR * grad_b2
            w3 -= LR * grad_w3
            b3 -= LR * grad_b3

            # Batch loss calculation (For logging)
            epoch_loss += np.square(y_pred - y_batch).sum()

        # Calculate average MSE
        epoch_mse = epoch_loss / X_train.shape[0]

        # Logging: Print only in Mode 1 and at specific intervals
        if mode_selection == 1 and t % 100 == 0:
            l2_penalty = (LAMBDA_L2 / 2) * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)))
            total_loss = epoch_mse + l2_penalty
            print(f"Epoch {t:4d} | Total Loss: {total_loss:.6f} (MSE: {epoch_mse:.6f})")

    # ==========================================
    # 7. PREDICTION AND REPORTING
    # ==========================================
    last_window_raw = data_values[-WINDOW_SIZE:].copy()
    reference_price_last = last_window_raw[-1, 3]

    for col_idx in [0, 1, 2, 3, 4, 5]:
        last_window_raw[:, col_idx] = (last_window_raw[:, col_idx] / reference_price_last) - 1.0

    last_window_raw[:, 6] = last_window_raw[:, 6] / (np.max(last_window_raw[:, 6]) + 1e-9)
    last_window_raw[:, 7] = last_window_raw[:, 7] / 100.0

    input_vector = last_window_raw.flatten().reshape(1, -1)

    h1_val = np.maximum(0, input_vector.dot(w1) + b1)
    h2_val = np.maximum(0, h1_val.dot(w2) + b2)
    predicted_ratio = h2_val.dot(w3) + b3
    predicted_change = predicted_ratio[0][0]

    predicted_price = reference_price_last * (1 + predicted_change)
    predicted_pct = predicted_change * 100

    print(f"Model Expectation: {predicted_pct:.2f}% (Price: {predicted_price:.2f})")

    if mode_selection == 1:
        print("\n" + "=" * 40)
        print("   RESULT REPORT")
        print("=" * 40)
        print(f"Reference Price (Last Day) : {reference_price_last:.2f}")
        print(f"Expected Change            : {predicted_pct:.2f}%")
        print(f"MODEL PREDICTION           : {predicted_price:.2f}")

        if ground_truth_available:
            print("-" * 30)
            print(f"Actual Date                : {reference_date}")
            print(f"Actual Price               : {actual_price:.2f}")
            error_percentage = (abs(predicted_price - actual_price) / actual_price) * 100
            model_direction = "UP" if predicted_price > reference_price_last else "DOWN"
            actual_direction = "UP" if actual_price > reference_price_last else "DOWN"

            direction_status = "✅ DIRECTION CORRECT" if model_direction == actual_direction else "❌ DIRECTION WRONG"
            success_status = "✅ SUCCESSFUL" if error_percentage < 5 else "⚠️ DEVIATION DETECTED"

            print(f"Error Rate                 : {error_percentage:.2f}%")
            print(f"General Status             : {success_status}")
            print(f"Direction Prediction       : {direction_status}")

    elif mode_selection == 2:
        # Add those exceeding the threshold to the list
        if predicted_pct >= threshold:
            recommended_stocks.append({
                'Stock': file_name.replace('.xlsx', ''),
                'Current': reference_price_last,
                'Prediction': predicted_price,
                'Increase': predicted_pct
            })

# ==========================================
# 8. BULK RESULTS SCREEN (Only MODE 2)
# ==========================================
if mode_selection == 2:
    print("\n" + "★" * 60)
    print(f"   SCAN RESULTS (Target Minimum Increase: {threshold}%)")
    print("★" * 60)

    if recommended_stocks:
        # Sort from the highest expected increase to the lowest
        recommended_stocks.sort(key=lambda x: x['Increase'], reverse=True)

        print(f"{'STOCK':<15} | {'CURRENT PRICE':<15} | {'TARGET PRICE':<15} | {'EXPECTED INCREASE'}")
        print("-" * 60)
        for s in recommended_stocks:
            print(f"{s['Stock']:<15} | {s['Current']:<15.2f} | {s['Prediction']:<15.2f} | {s['Increase']:.2f}%")
    else:
        print(f"\n⚠️ No stocks expected to increase above the {threshold}% threshold were found in the specified folder.")