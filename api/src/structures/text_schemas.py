from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Optional


class PhonemeRequest(BaseModel):
    text: str
    language: str = "a"  # Default to American English


class PhonemeResponse(BaseModel):
    phonemes: str
    tokens: list[int]


class StitchOptions(BaseModel):
    """Options for stitching audio chunks together"""
    gap_method: str = Field(
        default="static_trim",
        description="Method to handle gaps between chunks. Currently only 'static_trim' supported."
    )
    trim_ms: int = Field(
        default=0,
        ge=0,
        description="Milliseconds to trim from chunk boundaries when using static_trim"
    )

    @field_validator('gap_method')
    @classmethod
    def validate_gap_method(cls, v: str) -> str:
        if v != 'static_trim':
            raise ValueError("Currently only 'static_trim' gap method is supported")
        return v


class GenerateFromPhonemesRequest(BaseModel):
    phonemes: List[str]
    voice: str = Field(..., description="Voice ID to use for generation")
    trim_ms: int = Field(default=0, ge=0, le=100000, description="Trim milliseconds of audio before adding pause")
    pause_duration: float = Field(default=0.0, ge=0.0, le=60.0, description="Pause duration in seconds between sentences")
    speed: float = Field(default=1.0, ge=0.1, le=5.0, description="Speed factor for generation")
