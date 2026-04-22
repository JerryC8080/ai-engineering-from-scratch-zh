---
name: whisper-deployer
description: Pick Whisper variant, decoding params, VAD, chunking, and LoRA fine-tune plan for a given ASR workload.
version: 1.0.0
phase: 6
lesson: 05
tags: [asr, whisper, vad, lora, production]
---

Given the task (language(s), audio length, domain, latency target, hardware), output:

1. Model variant. whisper-tiny · small · medium · large-v3 · large-v3-turbo · distil-whisper · whisper.cpp (ggml). One-sentence reason.
2. Decode params. `language=` forced (ISO), `task=` transcribe|translate, `temperature=[0.0, 0.2, 0.4, 0.6]` fallback, `condition_on_previous_text=False` beyond chunk 1, `no_speech_threshold=0.6`.
3. Chunking + VAD. `chunk_length_s=30`, `stride=5`, Silero VAD threshold 0.5, min speech 250 ms, gap merge 500 ms. Always VAD if clip contains silence.
4. Fine-tune decision. None · prompt-only · LoRA r=16 · LoRA r=32 · full. Reason tied to domain-vocab size and available labeled hours.
5. Post-processing. Whisper-normalizer (English), multilingual-normalizer, forced alignment (wav2vec2) for &lt;50 ms timestamps, sentence re-boundary.

Refuse to ship Whisper without VAD when the input can contain silence — "Thanks for watching" hallucinations are the production-embarrassment default. Refuse `task="translate"` with `large-v3-turbo` — the turbo decoder was not trained on translation and emits garbage. Refuse `condition_on_previous_text=True` on long streams — causes drift loops. Flag any LoRA fine-tune on &lt;10 hours as risky for multilingual degradation.

Example input: "Live closed captions for a Spanish city-council meeting. Latency &lt; 3 s. Single RTX 4090."

Example output:
- Model: distil-whisper-large-v3 (Spanish fine-tune). Fits on 4090, closer to real-time than full large-v3-turbo.
- Decode: `language="es"`, `task="transcribe"`, `temperature=0.0`, `condition_on_previous_text=False`, `no_speech_threshold=0.6`.
- Chunking + VAD: Silero VAD @ 0.5; chunk 30s / stride 5s; always VAD — crowd noise during votes triggers hallucinations without it.
- Fine-tune: LoRA r=32 on 20 hours of Spanish municipal-meeting audio (contains jargon like "moción", "acta", "ad honorem"). Expect WER 8.5% → 4.2%.
- Post: multilingual Whisper-normalizer → wav2vec2-xlsr Spanish forced alignment for word-level timestamps → display.
