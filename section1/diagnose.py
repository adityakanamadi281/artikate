from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosisEntry:
    title: str
    investigated_first: str
    ruled_out: str
    root_cause: str
    fix: str

    def to_markdown(self) -> str:
        return (
            f"## {self.title}\n\n"
            f"**Investigated first:** {self.investigated_first}\n\n"
            f"**Ruled out:** {self.ruled_out}\n\n"
            f"**Root cause:** {self.root_cause}\n\n"
            f"**Concrete fix:** {self.fix}\n"
        )


ENTRIES = [
    DiagnosisEntry(
        title="Problem 1: Hallucinated Pricing",
        investigated_first=(
            "Whether the pricing answer was backed by fresh retrieved data from the source of truth "
            "or whether the model was answering from stale parametric memory."
        ),
        ruled_out=(
            "Pure temperature randomness, because repeated runs with the same prompt can still return "
            "the same wrong price if retrieval is missing or stale."
        ),
        root_cause=(
            "Retrieval failure for dynamic pricing data. The model should not be trusted to remember "
            "current prices without an authoritative retrieval path."
        ),
        fix=(
            "Use structured pricing retrieval, attach freshness metadata, require evidence before answering, "
            "and refuse if no current pricing record is retrieved."
        ),
    ),
    DiagnosisEntry(
        title="Problem 2: Language Switching",
        investigated_first=(
            "Instruction precedence in a system-prompt-plus-user-message architecture."
        ),
        ruled_out=(
            "Model incapability in Hindi or Arabic. GPT-4o supports both well; the issue is control flow, "
            "not raw language ability."
        ),
        root_cause=(
            "The system prompt is written in English but does not explicitly force replies to match the user's "
            "latest language, so the model sometimes defaults to English."
        ),
        fix=(
            "Add a system rule that always responds in the same language as the user's latest message, "
            "unless the latest message is in English."
        ),
    ),
    DiagnosisEntry(
        title="Problem 3: Latency Degradation",
        investigated_first=(
            "Queueing, worker saturation, retrieval growth, and external API latency because those can worsen "
            "gradually over time with no code changes."
        ),
        ruled_out=(
            "A prompt or model change after launch, because the scenario explicitly says none were made."
        ),
        root_cause=(
            "Most likely capacity pressure combined with larger retrieval workloads and possibly rate-limit "
            "backoff as the user base grew."
        ),
        fix=(
            "Add tracing for queue wait, retrieval time, prompt build time, and model latency; then cap context, "
            "optimize indexing, and scale concurrency."
        ),
    ),
]


def build_report() -> str:
    header = "# LLM Pipeline Diagnosis Log\n\n"
    body = "\n".join(entry.to_markdown() for entry in ENTRIES)
    return header + body


if __name__ == "__main__":
    print(build_report())
