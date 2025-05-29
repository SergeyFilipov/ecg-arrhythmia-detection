🫀 ECG Arrhythmia Detection with Deep Learning
A deep learning pipeline for automatic classification of cardiac arrhythmias using ECG signals from the MIT-BIH Arrhythmia Database.
The model combines signal denoising with wavelet transforms and a multi-layer CNN architecture, achieving over 98% test accuracy across 15 heartbeat classes.

Real-time ready. Clinically inspired. Built with ❤️, Python, and TensorFlow.


📁 1. Dataset: MIT-BIH Arrhythmia Database
The project uses ECG signals from the mitdb/ folder, which contains .dat and .atr files from the MIT-BIH Arrhythmia Database.

Signals are loaded using the wfdb library.

Only 15 clinically meaningful heartbeat symbols are retained.

Each ECG segment is sliced to 256 samples, centered around the detected QRS complex.

📌 Retained heartbeat classes:
['N', 'A', 'V', 'R', 'L', 'a', '!', 'F', 'f', 'j', 'J', '/', 'E', 'x', 'e']

🔍 Code: get_signal_data(), get_instances()


🌀 2. Signal Preprocessing with Wavelets
To reduce noise and preserve key signal structures, we apply wavelet-based denoising:

Wavelet: Symlet-7

Decomposition: 8 levels

Thresholding: MAD (Median Absolute Deviation)

Transformation: pywt.dwt(signal, 'sym7')

🔧 Code: apply_wavelet()

This process enhances signal clarity and suppresses high-frequency noise, improving model performance.


📊 3. Visualization of Raw and Filtered ECG
You can visualize ECG signals before and after preprocessing using:
plot_signal('100', start=0, end=1000, save_fig=True)

Results are saved automatically to the plots/ folder.

Great for debugging and inspecting waveform patterns.

🖼️ Code: plot_signal()
📁 Output: plots/


🧠 4. CNN Architecture for ECG Classification
The model uses dilated temporal convolutions to capture long-range dependencies between waveform peaks:

7 Conv1D layers with dilation rates: [1, 2, 4, 8, 16, 32, 64]

Global max pooling layer

Dense layers: 256 → 128 → 15 (Softmax)

This architecture balances computational efficiency with depth for real-time readiness.

🧩 Code: get_model()
💾 Saved model: model.keras


🏋️‍♀️ 5. Model Training & Evaluation
Training configuration:

Optimizer: Adam

Loss: Categorical Crossentropy

Callback: EarlyStopping (patience = 5)

After training:

Training curves (accuracy/loss) are saved as training_metrics.png

Evaluation is performed using a confusion matrix (absolute + normalized)

📈 Code: training loop model.fit()
📊 Evaluation: confusion_matrix()


🔍 6. Misclassification Analysis
To inspect errors, we visualize misclassified ECG segments for each class:

Randomly selected 5 misclassified signals per class

Facilitates interpretation of confusion patterns

🎯 Code: for i, n in enumerate(names):


🚀 7. How to Run
✅ Install dependencies
pip install -r requirements.txt

▶️ Run the full pipeline
python ecg_arrhythmia_classifier.py

💡 Or try the interactive version
jupyter notebook ecg_arrhythmia_classifier.ipynb


📦 Project Structure
File/Folder	Description
ecg_arrhythmia_classifier.py	Full model pipeline – from data to prediction
model.keras	Trained Keras model for direct use
plots/	ECG and training plots
mitdb/	Folder containing ECG signal data


📜 License
This project is released under the MIT License.
Feel free to use, fork, and adapt – but cite the original repository!

