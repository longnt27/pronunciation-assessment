import os
import sys
import time
import sounddevice as sd
import soundfile as sf
import io

from server.services.gop_service import GOPEvaluator

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ctcgop")
DURATION = 3.0
SAMPLE_RATE = 16000

TEST_PHRASES = [
    "HELLO",
    "BANANA",
    "COMPUTER",
    "AGRICULTURALIZATION"
]


def record_audio(duration):
    print("🎙  Đang thu âm...", end="\r")

    # Lấy thông tin thiết bị mặc định
    device_info = sd.query_devices(kind='input')
    native_sr = int(device_info['default_samplerate'])

    # Thu âm với Sample Rate gốc của máy (để tránh lỗi CoreAudio)
    recording = sd.rec(int(duration * native_sr), samplerate=native_sr, channels=1, dtype='float32')
    sd.wait()

    print("✅ Thu xong! Đang xử lý...        ")

    # Resample về 16000Hz cho model nó hiểu
    # Dùng librosa để resample chuẩn
    import librosa
    y = recording.flatten()
    if native_sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=SAMPLE_RATE)

    return y


def numpy_to_wav_bytes(audio_data):
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, SAMPLE_RATE, format='WAV')
    buffer.seek(0)
    return buffer.read()


def main():
    # Khởi tạo GOPEvaluator (tự động fallback nếu checkpoint local thiếu weights)
    evaluator = GOPEvaluator(MODEL_PATH if os.path.exists(MODEL_PATH) else None)

    print("\n" + "="*60)
    print("   TEST GOP - PHONEME LEVEL (GOP-CTC-AF-SDI)")
    print("="*60)
    print(f"Model: {MODEL_PATH}")

    for text in TEST_PHRASES:
        input(f"\n👉 Nhấn ENTER để đọc to từ: '{text}'")

        # Đếm ngược
        for i in range(3, 0, -1):
            print(f" {i}...", end="\r")
            time.sleep(0.5)

        audio_data = record_audio(DURATION)
        audio_bytes = numpy_to_wav_bytes(audio_data)

        # Chấm điểm
        start_t = time.time()
        result = evaluator.infer_gop(audio_bytes, text)
        end_t = time.time()

        if "error" in result:
            print(f"❌ Lỗi: {result['error']}")
            continue

        # Hiển thị kết quả
        print(f"⏱  Xử lý: {end_t - start_t:.2f}s")
        print(f"🗣  Phonemes: {result['transcript_phonemes']}")
        print(f"⭐ GOP Score: {result['average_gop']:.4f}")
        print("-" * 55)
        print(f"{'PHONEME':<10} | {'SCORE':<10} | {'CONFIDENCE':<12} | {'ĐÁNH GIÁ'}")
        print("-" * 55)

        for key, val in result['details'].items():
            ph = val['phoneme']
            score = val['gop_score']
            conf = val['confidence_score']

            # Thang điểm GOP:
            # > -0.5: Tốt
            # -0.5 đến -2.0: Khá/Trung bình
            # < -2.0: Tệ
            if score > -0.5:
                grade = "🟢 Tốt"
            elif score > -2.0:
                grade = "🟡 Khá"
            else:
                grade = "🔴 Tệ"

            print(f"{ph:<10} | {score:<10} | {conf:>6.1f}%      | {grade}")

    print("\n🎉 Xong bài test!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThoát.")
