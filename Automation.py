import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Convert ADFA text files to log files
def convert_to_logs(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".txt"):
                source_file = os.path.join(root, filename)
                target_file = os.path.join(output_dir, filename.replace(".txt", ".log"))
                shutil.copyfile(source_file, target_file)
                print(f"Converted {source_file} to {target_file}")

# Step 2: Preprocess ADFA LD data
def preprocess_adfa_ld(data_dir, output_dir):
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(output_dir, os.path.relpath(src_file, data_dir))
                dest_dir = os.path.dirname(dest_file)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

# Step 3: Read and process log files
def read_log_files(log_directory):
    if os.path.exists(log_directory):
        for filename in os.listdir(log_directory):
            if filename.endswith(".log"):  # Check if the file ends with .log
                file_path = os.path.join(log_directory, filename)
                with open(file_path, 'r') as file:
                    for line in file:
                        print(line)
    else:
        print(f"Directory '{log_directory}' does not exist.")

# Step 4: Read and concatenate NetFlow files
def read_netflow_files(directory):
    data_frames = []
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if directory doesn't exist

    print(f"Reading files from directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            print(f"Reading file: {filepath}")
            df = pd.read_csv(filepath, delimiter="\t", header=None)  # Adjust delimiter and header as necessary
            data_frames.append(df)

    if not data_frames:
        print("No .txt files found in the directory.")
        return pd.DataFrame()  # Return an empty DataFrame if no files are found

    return pd.concat(data_frames, ignore_index=True)

# Step 5: Process data for suspicious activity
def process_data(df):
    print("Data Overview:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())

    df['is_suspicious'] = df[1] > 10000  # Custom rule for identifying large transfers
    suspicious_traffic = df[df['is_suspicious']]

    print("\nSuspicious Traffic:")
    print(suspicious_traffic)

# Step 6: Parse and analyze OSSEC alerts
def parse_alerts(alerts_log):
    if not os.path.exists(alerts_log):
        print(f"Alerts log file '{alerts_log}' does not exist.")
        return []

    with open(alerts_log, 'r') as file:
        lines = file.readlines()

    print("Lines read from log file:")
    print(lines)

    alerts = [line.strip() for line in lines if "OSSEC" in line]
    return alerts

def analyze_alerts(alerts):
    df = pd.DataFrame(alerts, columns=["Alert"])
    df['Type'] = df['Alert'].apply(lambda x: 'Attack' if 'attack' in x.lower() else 'Normal')

    print("DataFrame:")
    print(df)

    alert_counts = df['Type'].value_counts()

    print("Alert counts:")
    print(alert_counts)

    if not alert_counts.empty:
        alert_counts.plot(kind='bar')
        plt.title('OSSEC Alert Counts')
        plt.xlabel('Alert Type')
        plt.ylabel('Count')
        plt.show()
    else:
        print("No alerts to plot.")

# Step 7: Evaluate IDS performance
def evaluate_ids(train_data, train_labels, test_data, test_labels):
    # Replace with actual IDS training and prediction
    ids_predictions = np.random.randint(2, size=len(test_labels))  # Example random predictions

    cm = confusion_matrix(test_labels, ids_predictions)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = precision_score(test_labels, ids_predictions)
    recall = recall_score(test_labels, ids_predictions)
    f1 = f1_score(test_labels, ids_predictions)
    accuracy = accuracy_score(test_labels, ids_predictions)

    return tpr, fpr, precision, recall, f1, accuracy

# Step 8: Evaluate performance metrics from OSSEC alerts
def parse_ossec_alerts(log_file):
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' does not exist.")
        return []

    alerts = []
    with open(log_file, 'r') as file:
        for line in file:
            alerts.append(line.strip())
    return alerts

def evaluate_performance(alerts):
    true_positives = sum(1 for alert in alerts if "attack detected" in alert)
    false_positives = sum(1 for alert in alerts if "false alarm" in alert)
    false_negatives = sum(1 for alert in alerts if "missed attack" in alert)
    total_attacks = true_positives + false_negatives
    total_normal = false_positives + sum(1 for alert in alerts if "normal" in alert)

    detection_rate = true_positives / total_attacks if total_attacks else 0
    false_positive_rate = false_positives / total_normal if total_normal else 0
    false_negative_rate = false_negatives / total_attacks if total_attacks else 0
    false_alarm_rate = false_positives / (false_positives + true_positives) if (false_positives + true_positives) else 0

    return {
        "Detection Rate (DR)": detection_rate,
        "False Positive Rate (FPR)": false_positive_rate,
        "False Negative Rate (FNR)": false_negative_rate,
        "False Alarm Rate (FAR)": false_alarm_rate
    }

if __name__ == "__main__":
    # Convert ADFA text files to log files
    training_data_dir = '/home/user/adfa-ld/training_data'
    validation_data_dir = '/home/user/adfa-ld/validation_data'
    attack_data_dir = '/home/user/adfa-ld/attack_data'

    convert_to_logs(training_data_dir, '/var/log/adfa-la-training')
    convert_to_logs(validation_data_dir, '/var/log/adfa-la-validation')
    convert_to_logs(attack_data_dir, '/var/log/adfa-la-attack')

    # Preprocess ADFA LD data
    base_dir = "/home/user/adfa-ld"
    preprocess_adfa_ld(os.path.join(base_dir, "attack_data"), os.path.join(base_dir, "preprocessed/attack_data"))
    preprocess_adfa_ld(os.path.join(base_dir, "training_data"), os.path.join(base_dir, "preprocessed/training_data"))
    preprocess_adfa_ld(os.path.join(base_dir, "validation_data"), os.path.join(base_dir, "preprocessed/validation_data"))

    # Read and process log files
    log_directory = '/var/log/adfa-la-training'  # Ensure this path is correct
    read_log_files(log_directory)

    # Read and process NetFlow data
    netflow_directory = "/home/user/adfa-la/netflow_ids_label/netflow_ids_label/Training_Data_Master"
    df = read_netflow_files(netflow_directory)
    if not df.empty:
        print(df.head())  # Print the first few rows of the concatenated DataFrame to verify
        process_data(df)
    else:
        print("No data to process.")

    # Analyze OSSEC alerts
    alerts_log = '/var/ossec/logs/alerts/alerts.log'
    alerts = parse_alerts(alerts_log)
    if alerts:
        analyze_alerts(alerts)
    else:
        print("No alerts to analyze.")

    # Evaluate IDS performance
    data = np.random.randn(1000, 10)  # Example data matrix
    labels = np.random.randint(2, size=1000)  # Example binary labels

    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    tprs, fprs, precisions, recalls, f1s, accuracies = [], [], [], [], [], []

    for train_index, test_index in skf.split(data, labels):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        tpr, fpr, precision, recall, f1, accuracy = evaluate_ids(train_data, train_labels, test_data, test_labels)
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)

    mean_tpr = np.mean(tprs)
    mean_fpr = np.mean(fprs)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    mean_accuracy = np.mean(accuracies)

    print("Mean True Positive Rate (TPR):", mean_tpr)
    print("Mean False Positive Rate (FPR):", mean_fpr)
    print("Mean Precision:", mean_precision)
    print("Mean Recall:", mean_recall)
    print("Mean F1 Score:", mean_f1)
    print("Mean Accuracy:", mean_accuracy)

    # Evaluate performance metrics from OSSEC alerts
    alerts = parse_ossec_alerts(alerts_log)
    if alerts:
        metrics = evaluate_performance(alerts)
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        print("No alerts to evaluate.")
