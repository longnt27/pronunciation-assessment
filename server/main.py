import uvicorn
import os
import io
import librosa
import numpy as np
import re
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from g2p_en import G2p
from server.utils.audio_processor import preprocess_audio

# Import Services
from server.services.stress_service import StressEvaluator
from server.services.gop_service import GOPEvaluator

app = FastAPI(
    title="Pronunciation Assessment API",
    description="API chấm điểm phát âm (GOP) và bắt lỗi trọng âm (Stress)",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_MODEL_PATH = os.path.join(BASE_DIR, "models", "sylstress", "stress_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "sylstress", "scaler_params.json")
GOP_MODEL_PATH = os.path.join(BASE_DIR, "models", "ctcgop")

# Global instances
stress_service = None
gop_service = None
g2p = G2p()


@app.on_event("startup")
async def startup_event():
    global stress_service, gop_service
    print("🚀 Starting up API...")

    # Init Services
    if os.path.exists(STRESS_MODEL_PATH):
        stress_service = StressEvaluator(STRESS_MODEL_PATH, SCALER_PATH)
    else:
        print(f"⚠ Warning: Không tìm thấy Stress Model tại {STRESS_MODEL_PATH}")

    if os.path.exists(GOP_MODEL_PATH):
        gop_service = GOPEvaluator(GOP_MODEL_PATH)
    else:
        print(f"⚠ Warning: Không tìm thấy GOP Model tại {GOP_MODEL_PATH}")


def get_truth_stress(word):
    """Lấy trọng âm chuẩn từ từ điển CMU"""
    # G2P trả về: ['B', 'AH0', 'N', 'AE1', 'N', 'AH0']
    phonemes = g2p(word)
    # Lọc lấy số (stress marker)
    stress_seq = []
    for p in phonemes:
        if any(char.isdigit() for char in p):
            val = int(re.search(r'\d+', p).group())
            # Quy ước: 1 (Primary) -> 1, còn lại (0, 2) -> 0
            stress_seq.append(1 if val == 1 else 0)
    return stress_seq


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "gop_service": gop_service is not None and gop_service.model is not None,
        "stress_service": stress_service is not None,
        "stress_model_loaded": stress_service.model is not None if stress_service else False
    }


@app.post("/assess")
async def assess_pronunciation(
    word: str = Form(..., description="Từ cần chấm điểm (vd: 'banana')"),
    audio: UploadFile = File(...),
    method: str = Form(
        "ctc_viterbi",
        description="Embedded CTC-Viterbi alignment and phone LPP scoring",
    ),
):
    """
    Endpoint All-in-One:
    - Input: Audio + Word (+ optional method)
    - Output: Goodness of Pronunciation (Phoneme) + Syllable Stress Detection
    """
    if not stress_service or not gop_service:
        raise HTTPException(503, "Services chưa sẵn sàng. Check log server.")

    try:
        # 1. Đọc và chuẩn hóa audio
        audio_bytes = await audio.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        y_normalized = preprocess_audio(y, sr=sr)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, y_normalized, sr, format='WAV')
        wav_buffer.seek(0)
        clean_audio_bytes = wav_buffer.read()

        # 2. One CTC forward pass supplies both Viterbi phone regions and scores.
        gop_result = gop_service.infer_gop(clean_audio_bytes, word, method=method)

        if "error" in gop_result:
            raise HTTPException(500, f"GOP Error: {gop_result['error']}")

        # 3. CHẠY STRESS DETECTION VỚI BIÊN GIỚI TỪ GOP
        stress_result = stress_service.predict(
            clean_audio_bytes,
            word,
            alignments=gop_result.get("alignment"),
            method=gop_result.get("timing_method", method),
        )

        if "error" in stress_result:
            raise HTTPException(500, f"Stress Error: {stress_result['error']}")

        # 4. TỔNG HỢP KẾT QUẢ
        # GOP Scores (Map sang format chi tiết cho UI)
        phones_score = {}
        for k, v in gop_result['details'].items():
            # Trả về gop_score (LPR) cho UI
            phones_score[k] = v['gop_score']

        # Overall Score
        if "overall_score" in gop_result and gop_result["overall_score"] is not None:
            overall_score = float(gop_result["overall_score"])
        else:
            avg_gop = gop_result.get('average_gop', 0)
            overall_score = max(0.0, min(100.0, float(np.exp(avg_gop) * 100)))

        # Stress Processing
        truth_stress = stress_result.get("truth", get_truth_stress(word))
        pred_stress = stress_result.get("infer", [0] * len(truth_stress))
        stress_conf = float(stress_result.get("confidence", stress_result.get("stress_probability", 0.8)))

        # Đảm bảo độ dài match với truth
        target_len = len(truth_stress)
        if len(pred_stress) > target_len:
            pred_stress = pred_stress[:target_len]
        elif len(pred_stress) < target_len:
            pred_stress = pred_stress + [0] * (target_len - len(pred_stress))

        return {
            "status": "success",
            "word": word,
            "method": gop_result.get("method", method),
            "scoring_method": gop_result.get("scoring_method", method),
            "timing_method": gop_result.get("timing_method", method),
            "external_aligner_used": gop_result.get("external_aligner_used", False),
            "canonical_sequence_constrained": gop_result.get(
                "canonical_sequence_constrained", True
            ),
            "phones": phones_score,
            "stress": {
                "truth": truth_stress,
                "infer": pred_stress,
                "confidence": round(stress_conf, 3),
                "syllable_count": target_len
            },
            "overall_score": round(overall_score, 1),
            "details": {
                "phoneme_details": gop_result.get("details", {}),
                "syllable_details": stress_result.get("syllables", []),
                "speech_bounds": gop_result.get("speech_bounds", {}),
                "latency_ms": gop_result.get("latency_ms", {}),
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
