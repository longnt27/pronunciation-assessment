import librosa
import noisereduce as nr
import numpy as np


def preprocess_audio(y, sr=16000):
    y_denoised = y

# 2. CẮT KHOẢNG LẶNG (TRIMMING) - CHIẾN THUẬT "VÉT CẠN"

    # top_db=60: Cực nhạy. Chỉ cắt những gì nhỏ hơn 1/1000000 năng lượng đỉnh.
    # frame_length=1024: Cửa sổ trung bình, không quá nhỏ để tránh noise gai.
    # Dùng trim() thay vì split() để chỉ cắt đầu/đuôi, không cắt giữa.
    y_trimmed, index = librosa.effects.trim(y_denoised, top_db=35, frame_length=1024, hop_length=256)

    # Lấy chỉ số cắt từ librosa
    start_idx = index[0]
    end_idx = index[1]

    # --- BACKTRACKING (CỨU VIỆN) ---
    # Lùi lại 400ms (0.4s) từ điểm librosa phát hiện.
    # Chấp nhận lấy thêm tí noise đầu còn hơn mất chữ D.
    pad_start = int(0.2 * sr)
    pad_end = int(0.2 * sr)

    final_start = max(0, start_idx - pad_start)
    final_end = min(len(y_denoised), end_idx + pad_end)

    y_final = y_denoised[final_start:final_end]

    # 3. ĐỘN (MINIMUM DURATION)
    MIN_DURATION = 1.0 # giây
    min_samples = int(MIN_DURATION * sr)

    if len(y_final) < min_samples:
        missing = min_samples - len(y_final)
        pad_left = missing // 2
        pad_right = missing - pad_left
        y_padded = np.pad(y_final, (pad_left, pad_right), mode='constant')
    else:
        y_padded = y_final

    # 4. NORMALIZE
    y_normalized = librosa.util.normalize(y_padded)

    return y_normalized
