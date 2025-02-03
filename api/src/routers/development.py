from typing import List

import numpy as np
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from loguru import logger

from ..services.audio import AudioService, AudioNormalizer
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..services.text_processing import phonemize, smart_split
from ..services.text_processing.vocabulary import tokenize
from ..services.tts_service import TTSService
from ..structures.text_schemas import (
    GenerateFromPhonemesRequest,
    PhonemeRequest,
    PhonemeResponse,
)

router = APIRouter(tags=["text processing"])


async def get_tts_service() -> TTSService:
    """Dependency to get TTSService instance"""
    return await TTSService.create()  # Create service with properly initialized managers


@router.post("/dev/phonemize", response_model=PhonemeResponse)
async def phonemize_text(request: PhonemeRequest) -> PhonemeResponse:
    """Convert text to phonemes and tokens

    Args:
        request: Request containing text and language
        tts_service: Injected TTSService instance

    Returns:
        Phonemes and token IDs
    """
    try:
        if not request.text:
            raise ValueError("Text cannot be empty")

        # Get phonemes
        phonemes = phonemize(request.text, request.language)
        if not phonemes:
            raise ValueError("Failed to generate phonemes")

        # Get tokens (without adding start/end tokens to match process_text behavior)
        tokens = tokenize(phonemes)
        return PhonemeResponse(phonemes=phonemes, tokens=tokens)
    except ValueError as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error in phoneme generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail={"error": "Server error", "message": str(e)}
        )


@router.post("/text/generate_from_phonemes", tags=["deprecated"])
@router.post("/dev/generate_from_phonemes")
async def generate_from_phonemes(
    request: GenerateFromPhonemesRequest,
    tts_service: TTSService = Depends(get_tts_service)
) -> Response:
    """Generate audio directly from phonemes

    Args:
        request: Request containing phonemes and generation parameters
        tts_service: Injected TTSService instance

    Returns:
        WAV audio bytes
    """
    # Validate phonemes first
    if not request.phonemes or len(request.phonemes) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": "Phonemes cannot be empty"}
        )

    try:
        n: List[np.ndarray] = []
        sr: int = 24000
        trim_samples: int = 0
        if request.trim_ms > 0:
            trim_samples = int((request.trim_ms / 1000) * sr)
        pause_duration = np.zeros(int(sr * request.pause_duration), dtype=np.float32)
        for i, phonemes in enumerate(request.phonemes):
            audio, _ = await tts_service.generate_from_phonemes(phonemes=phonemes, voice=request.voice, speed=request.speed)
            if trim_samples > 0:
                audio = audio[trim_samples:-trim_samples]
            n.append(audio)
            if i < len(n):
                n.append(pause_duration)
        audio = np.concatenate(n) if len(n) > 1 else n[0]
        content = await AudioService.convert_audio(
            audio, 24000, "wav",
            is_first_chunk=True,
            is_last_chunk=True
        )
        return Response(
            content=content,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "Cache-Control": "no-cache",
            }
        )

    except ValueError as e:
        logger.error(f"Invalid request: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid request", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Server error", "message": str(e)}
        )
