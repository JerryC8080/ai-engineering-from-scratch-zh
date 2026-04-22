---
name: tts-picker
description: Pick a TTS model, voice, latency budget, cloning policy, and watermark strategy for a given workload.
version: 1.0.0
phase: 6
lesson: 07
tags: [tts, voice-cloning, watermark, kokoro, f5-tts, vibevoice, orpheus]
---

Given the task (language(s), domain, latency budget, cloning needs, license constraints, compute), output:

1. Model. Kokoro-82M · F5-TTS · Orpheus-3B · VibeVoice 1.5B/7B · Sesame CSM-1B · XTTS v2 · commercial (ElevenLabs / Inworld / Play.ht). One-sentence reason.
2. Voice. Built-in voice-pack (Kokoro) · cloned from reference (F5-TTS, VibeVoice, Orpheus) · proprietary library. For cloning: required reference-clip length, quality gate (SNR &gt; 20 dB, no music bed), consent documentation.
3. Frontend. Text normalizer (numbers, abbreviations, dates, currencies), phonemizer if multilingual, sentence splitter, optional SSML for emphasis/pauses. NVIDIA NeMo Text Normalizer is the production default.
4. Runtime. TTFA target, batch vs streaming, GPU vs CPU, quantization (8-bit / 4-bit). Kokoro + ONNX on CPU is the "free" path; F5-TTS needs a GPU.
5. Safety. Watermark (AudioSeal), consent policy for cloning, AI-disclaimer injection, rate-limit + monitoring for abuse signals. Refuse to ship cloning without at least AudioSeal.

Refuse voice cloning without a watermarking strategy — voice-based fraud is a 2026 regulatory exposure. Refuse Kokoro for voice cloning (not supported — has built-in voices only). Refuse F5-TTS for commercial deployments (CC-BY-NC license). Refuse "just use ElevenLabs" without checking per-minute cost at the deploy volume.

Example input: "Voice agent for an airline customer-service IVR. English + Spanish. &lt; 200 ms TTFA. 1M minutes/month. Must not clone customers."

Example output:
- Model: Kokoro-82M (EN), Orpheus-3B with Spanish adapter or Suno-Bark-multi for ES. Both Apache-2.0; neither clones customers (Kokoro can't, Orpheus used with built-in voices only).
- Voice: Kokoro `af_bella` (EN primary), `am_adam` (EN male fallback); Orpheus `esp_*` voices for ES. Lock to 2 voices per language; never accept reference audio from callers.
- Frontend: NVIDIA NeMo normalizer + espeak-ng phonemizer for ES (Spanish G2P improves prosody noticeably). SSML for pauses on account numbers.
- Runtime: Kokoro ONNX INT8 on CPU per call (≈ $0.001/min), Orpheus on shared A10G for ES only (≈ $0.006/min). Target 120 ms TTFA.
- Safety: AudioSeal on all output audio; pipeline refuses any "read this in &lt;celebrity&gt;'s voice" prompt injection. Log outputs 30 days for abuse monitoring.
