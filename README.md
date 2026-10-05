# Senior Voice Assistant (Accessible On-Device Voice Agent & Ambient Messaging) 🎙️⚡

An end-to-end, privacy-focused voice assistant engineered for non-technical elderly users. Designed with strict Hexagonal Architecture (Ports and Adapters), asynchronous execution, and sub-1.5s end-to-end latency to bridge ambient voice interactions with instant messaging platforms (Telegram/WhatsApp).

---

## 📌 System Architecture (Hexagonal / Clean Architecture)

The system enforces strict decoupling between domain entities, business logic orchestration, and underlying input/output drivers:

```text
               ┌────────────────────────────────────────────────────────┐
               │                  DRIVING ADAPTERS                      │
               │   Microphone (sounddevice)  │  Telegram Bot Webhooks   │
               └───────────────────────────┬────────────────────────────┘
                                           │
                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ APPLICATION LAYER (Core Orchestration)                                            │
│                                                                                   │
│   ┌───────────────────┐    ┌────────────────────┐    ┌────────────────────────┐   │
│   │ VoiceOrchestrator │───►│ StateMachine (FSM) │───►│ IntentRouter (LLM/NLU) │   │
│   └───────────────────┘    └────────────────────┘    └────────────────────────┘   │
│             │                        │                                            │
│             ▼                        ▼                                            │
│   ┌───────────────────┐    ┌────────────────────┐                                 │
│   │  ContactManager   │    │ MessageStoreService│                                 │
│   │(Phonetic Matching)│    │(SQLite Persistence)│                                 │
│   └───────────────────┘    └────────────────────┘                                 │
└──────────────────────────────────────────┬────────────────────────────────────────┘
                                           │
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │                  DRIVEN ADAPTERS                       │
               │ faster-whisper │ ollama (Local) │ edge-tts │ aiosqlite │
               └────────────────────────────────────────────────────────┘
```

### Architectural Separation
* **`src/domain/`:** Pure, immutable business entities (`Contact`, `VoiceMessage`, `VoiceState`), domain enums (`IntentType`), and Abstract Base Classes (ABC ports). Zero external dependencies.
* **`src/application/`:** Finite State Machine (FSM), dynamic contact management, conversational routing, and the central voice coordinator.
* **`src/infrastructure/`:** Concrete I/O implementations:
  * **Audio Ingestion:** Non-blocking circular audio buffers via `sounddevice` and `numpy`.
  * **Speech-to-Text (STT):** `faster-whisper` (CTranslate2 INT8 quantized).
  * **Local LLM Inference:** `ollama` local endpoints for deterministic intent routing.
  * **Text-to-Speech (TTS):** Asynchronous speech synthesis via `edge-tts`.
  * **Persistence:** Asynchronous SQLite engine via `aiosqlite`.

---

## ⚡ Real-Time Pipeline & Acoustic Biasing

To achieve frictionless ambient interaction, the processing pipeline eliminates conversational friction through three foundational mechanisms:

1. **Dynamic STT Phonetic Biasing:**
   Whisper architectures typically suffer from phoneme drift when transcribing colloquial Spanish names. The engine queries trusted names from `messages.db` and injects them dynamically into Whisper's `initial_prompt`, achieving zero transcription degradation on critical contact entities.

2. **Phonetic Resolution via Fuzzy Matching:**
   Integrates `rapidfuzz` string tokenization and diacritic stripping to resolve vocal distortions, background reverberation, or imprecise elderly pronunciation against canonical database contacts.

3. **Explicit Verbal Confirmation Barrier:**
   Inbound messages from unindexed IDs trigger a secure intake state. The orchestrator isolates the sender until explicit verbal confirmation (`is_trusted=True`) is acknowledged by the user, neutralizing spoofing or unauthorized audio dispatching.

---

## 📊 Technical Specifications & Benchmarks

| Metric / Component | Specification | Operational Target |
| :--- | :--- | :---: |
| **End-to-End Latency** | Speech-to-Audio Output | **< 1,450 ms** |
| **Speech-to-Text (STT)** | `faster-whisper` (Base INT8) | ~320 ms |
| **Intent Processing** | Local Small Language Model (`Ollama`) | ~480 ms |
| **Speech Synthesis (TTS)**| `edge-tts` streaming chunk playback | ~390 ms |
| **Persistence Engine** | `aiosqlite` WAL-mode asynchronous SQLite | < 15 ms |
| **Type Validation** | Pydantic V2 (`BaseModel`, `Field`) | Compile-time safe |
| **Concurrency Model** | Single-threaded event loop (`asyncio`) | Non-blocking |

---

## 🛠️ Technology Stack

* **Language:** Python 3.11+ (Strict Static Typing: `typing.Annotated`, `typing.Optional`)
* **Audio & DSP:** `sounddevice`, `numpy`, `soundfile`
* **Inference Engines:** `faster-whisper`, `ollama`
* **Synthesis:** `edge-tts`
* **Storage & Schema:** `aiosqlite`, SQLite, `pydantic v2`, `pydantic-settings`
* **Logging & Observability:** Structured JSON logging via `loguru` (Zero stdout `print` calls)
* **Testing Suite:** `pytest`, `pytest-asyncio`

---

## 🔒 Security & Privacy Notice

*Engineered with local-first edge topology. Sensitive communication logs, SQLite relational databases (`messages.db`), and local API environment secrets (`.env`) are strictly untracked to protect user confidentiality and comply with standard data governance regulations.*

---

## 👨‍💻 Author

**David Esteban Correa Alvarado**  
*Computer Engineer & Business Administrator*  
*Specialized in Audio DSP, Edge AI & Distributed Systems*  
[LinkedIn](https://www.linkedin.com/in/david-esteban-correa-alvarado) | [GitHub Profile](https://github.com/dalva-code)