// ============================================================================
// LLM Client Configuration for Resume Extraction
// ============================================================================
//
// Uses Groq (FREE tier) with Llama 3.3 70B as the primary model.
// Groq offers a generous free tier: 30 RPM, 14.4K tokens/min.
// Temperature is set to 0 for deterministic, reproducible outputs.
//
// BAML handles JSON schema enforcement automatically — the model is
// constrained to produce only valid JSON matching our type definitions.
//
// ─── HOW TO SWITCH MODELS ───────────────────────────────────────────
// This is the ONLY file you need to edit to change the LLM provider.
//
// To change the model:
//   1. Update the `provider` field  (openai, google-ai, anthropic, etc.)
//   2. Update the `model` name
//   3. Update the `api_key` env var name
//   4. For non-standard providers, update `base_url`
//   5. Run: baml-cli generate   (to regenerate the Python client)
//
// Popular free/cheap options:
//   Groq       → "llama-3.3-70b-versatile"  (FREE, 30 RPM — current)
//   Groq       → "llama-3.1-8b-instant"     (FREE, 30 RPM — faster)
//   google-ai  → "gemini-2.0-flash"         (FREE, 15 RPM, 1M TPM)
//   openai     → "gpt-4o-mini"              (paid, ~$0.15/1M input)
//   anthropic  → "claude-3-5-haiku-20241022"(paid, ~$0.25/1M input)
//
// Environment variable required: GROQ_API_KEY
//   Get free key at: https://console.groq.com/keys
// ============================================================================

// Primary client: Groq Llama 3.3 70B (free, high quality structured extraction)
client<llm> ResumeExtractionClient {
  provider openai-generic
  options {
    model "llama-3.3-70b-versatile"
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    temperature 0
    max_tokens 8000
  }
}

// Fallback client: Groq with smaller model
client<llm> ResumeExtractionFallback {
  provider openai-generic
  options {
    model "llama-3.1-8b-instant"
    base_url "https://api.groq.com/openai/v1"
    api_key env.GROQ_API_KEY
    temperature 0
    max_tokens 8000
  }
}

// ── Commented alternatives — uncomment to switch ──────────────────
//
// --- OpenAI (paid) ---
// client<llm> ResumeExtractionClient {
//   provider openai
//   options {
//     model "gpt-4o-mini"
//     temperature 0
//     api_key env.OPENAI_API_KEY
//   }
// }
//
// --- Anthropic (paid) ---
// client<llm> ResumeExtractionClient {
//   provider anthropic
//   options {
//     model "claude-3-5-haiku-20241022"
//     temperature 0
//     api_key env.ANTHROPIC_API_KEY
//   }
// }
//
// --- Groq Llama 3.1 8B (fastest, weakest) ---
// client<llm> ResumeExtractionClient {
//   provider openai-generic
//   options {
//     model "llama-3.1-8b-instant"
//     base_url "https://api.groq.com/openai/v1"
//     api_key env.GROQ_API_KEY
//     temperature 0
//   }
// }

// Gemini 2.0 Flash primary; Groq 70B fallback if rate-limited.
client<llm> ResumeExtractor {
  provider round-robin
  options {
    strategy [ResumeExtractionClient, ResumeExtractionFallback]
  }
}
