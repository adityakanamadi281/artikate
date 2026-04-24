# Section 1: Diagnose a Failing LLM Pipeline

## Problem 1: Confidently Wrong Answers About Product Pricing

### Diagnosis Log

Investigation started with a simple question: is the model answering from current product data or from its pretraining memory? I first compared wrong production answers against the authoritative pricing table in the source system. The failures were consistent with stale or missing retrieved facts, not random variation in wording. I then ruled out temperature as the primary cause by checking whether wrong answers were deterministic across repeated runs. If the same wrong price appears at low temperature and high temperature, that points away from sampling noise and toward missing grounding.

Next, I distinguished among the four candidate causes:

- Prompt issue: test whether the prompt explicitly requires use of retrieved pricing data and refusal when pricing is absent.
- Retrieval issue: inspect whether the correct pricing record is present in the retrieved context at all.
- Model temperature issue: replay the same query at `temperature=0` and compare factual stability.
- Knowledge cutoff issue: ask a pricing question with retrieval fully disabled. If the answer remains wrong and matches stale public pricing, the model is using outdated parametric knowledge.

Root cause identified: retrieval issue.

Why:

- Pricing is a fast-changing business fact, so the correct system design should never rely on model memory.
- In this failure pattern, the model answers confidently because it received either no pricing snippet, the wrong pricing snippet, or a stale cached result. The model is not the source of truth; the retrieval layer is.

Concrete fix:

1. Store pricing in a structured source of truth, not only in unstructured documents.
2. Add retrieval assertions for pricing intents: if no current pricing record is retrieved, refuse to answer.
3. Add freshness metadata and invalidate stale caches.
4. Update the prompt to require cited pricing evidence before answering.

## Problem 2: Responses Occasionally Switch to English for Hindi or Arabic Users

### Diagnosis Log

I investigated the message hierarchy first, because language drift in multilingual systems is usually caused by instruction precedence rather than model incapability. The likely architecture is a system prompt written in English plus a user message in Hindi or Arabic. When the system prompt says things like "You are a helpful assistant" but does not explicitly bind response language to the user input, the model often defaults to the dominant instruction language: English.

I ruled out model support as the cause because GPT-4o handles both Hindi and Arabic well. I also ruled out randomness as the main cause because the symptom is conditional: it happens more often when the user message contains mixed-language terms, copied product names, or short follow-ups like "why?" after an earlier English turn.

Root cause identified: the system prompt does not explicitly constrain output language, so the model follows the higher-priority English instruction context.

Specific prompt change:

```text
System:
You are a multilingual customer support assistant.
Always answer in the same language as the user's latest message.
Do not default to English unless the user's latest message is in English.
If the latest user message mixes languages, use the language that dominates the message and preserve product names, URLs, code, and proper nouns exactly as written.
If you are unsure which language dominates, ask a brief clarification question in the user's last-used language.
```

Why this works:

- It makes the language rule explicit at system level.
- It keys off the latest user message, which is testable.
- It is language-agnostic and does not hardcode Hindi or Arabic.

## Problem 3: Latency Degraded from 1.2s to 8-12s Over Two Weeks

### Diagnosis Log

I started with causes that can worsen over time even when application code does not change. The first branch was infrastructure saturation: request queueing, exhausted worker concurrency, database contention, and vector index growth. The second branch was external dependency drift: slower upstream model API response, network issues, or autoscaling lag in a managed service. The third branch was data growth: a larger prompt, larger retrieved context, or a vector store whose query time has increased as new documents accumulated.

I would investigate these causes first:

1. Traffic growth causing queueing or worker saturation.
2. Retrieval/index growth causing slower search and larger prompt assembly.
3. Upstream API latency or rate-limit backoff causing cumulative delay.

Why this order:

- The symptom worsened gradually over time with no code change, which is classic capacity pressure.
- Queueing and retrieval growth are both usage-linked and commonly invisible until launch scale.
- These causes can turn a 1-second service into a 10-second service without any prompt changes.

Likely root cause:

Capacity plus retrieval growth.

Concrete fix:

1. Add end-to-end tracing for retrieval time, prompt-build time, model latency, and queue wait time.
2. Cap retrieved tokens and summarize redundant chunks before generation.
3. Scale worker concurrency and connection pools based on observed throughput.
4. Rebuild or optimize the retrieval index and add caching for repeated queries.

## Post-Mortem Summary

The chatbot issues came from three different layers of the system rather than a single model failure. Pricing errors were caused by the bot answering without reliably grounding itself in the current pricing source, which made it sound confident even when the underlying data was missing or stale. The language issue came from the instruction setup: because the system prompt was written in English and did not explicitly require matching the user’s language, the model sometimes fell back to English. The latency increase was not caused by a prompt change; it was more consistent with normal production growth effects such as larger queues, slower retrieval as data volume increased, and longer waits on external model calls.

We can fix these issues with targeted changes. For pricing, the bot should only answer from a validated pricing source and refuse otherwise. For language consistency, the system prompt should explicitly require responses in the user’s latest language. For latency, we need better tracing, tighter context limits, and capacity tuning across retrieval and inference. Together, these changes improve factual reliability, multilingual consistency, and response speed without changing the underlying product experience.

# Section 3: Fine-Tuning DistilBERT

## Model Selection Justification

I chose a fine-tuned DistilBERT classifier rather than prompt-classifying with a large hosted LLM.

The decision is mainly about latency, throughput, and cost discipline:

- Incoming volume is one ticket every 30 seconds, which is 2,880 tickets per day.
- The latency budget is under 500ms on a single CPU server.
- DistilBERT inference on CPU with short ticket text and batch size 1-20 is realistically within tens to low hundreds of milliseconds per ticket on commodity hardware.

Rough calculation:

- If average inference is 120ms per ticket, daily compute time is `2,880 x 0.12 = 345.6 seconds`, or under 6 minutes of CPU inference time across the whole day.
- Even at 300ms per ticket, the system still fits the SLA comfortably because arrivals are 30 seconds apart.

Why not use a large prompt-based model:

- Remote API latency alone often lands in the 700ms to multi-second range before application overhead.
- Even if quality were acceptable, tail latency and network variability make a 500ms CPU-only SLA hard to guarantee.
- Cost also compounds: 2,880 tickets per day means over 1 million classifications per year for a task that is narrow and label-bounded.

For a five-class intent problem with 1,000 labeled examples, a small fine-tuned transformer is the better engineering fit: predictable, local, fast, and cheap.

## Most Confused Classes

The two classes most likely to be confused are `complaint` and `technical_issue`.

Why they are hard to separate:

- Many frustrated users describe a bug in emotional language, for example: "Your app keeps crashing and this is unacceptable."
- The semantic core can contain both a product malfunction and dissatisfaction.
- Surface tone alone is not enough because a complaint may be purely sentiment-driven, while a technical issue is operational and often reproducible.

What would improve separation:

1. More labeled borderline cases where both frustration and malfunction appear.
2. Extra signals such as whether the ticket includes device details, error messages, steps to reproduce, screenshots, or timestamps.
3. A secondary multi-label or hierarchical scheme where "issue type" and "sentiment/escalation level" are separated instead of forced into one label.

# Section 4: Written Systems Design Review

## Question A: Prompt Injection and LLM Security
* Prompt Ingestion:
Prompt ingestion is the structured process of handling user inputs in LLM systems.
It includes input collection, sanitization, validation, and intent classification.
Inputs are treated as untrusted and cleaned to prevent prompt injection attacks.
The system routes queries to appropriate workflows using controlled templates.
This ensures accuracy, scalability, and safe execution of LLM pipelines.

1. Input Sanitization:
Detect and block malicious instructions before they reach the model.
Prevents prompt injection attacks
Stops users from overriding system instructions
First line of defense in LLM pipelines

```py
def sanitize_input(text):
    forbidden = ["ignore previous", "you are now", "disregard above"]
    safe = text.replace("\n", " ").replace(":", ";").strip()
    
    for f in forbidden:
        if f in safe.lower():
            raise ValueError("Prompt injection detected")
    
    return safe
```

2. Secure Prompt Construction (Instruction Hierarchy):
Separate system instructions from user input and enforce strict control.
Prevents user override attacks
Maintains instruction priority (system > user)
Ensures consistent behavior

```py
def build_prompt(user_input):
    system_prompt = "You are a secure assistant. Follow only system instructions."
    sanitized_input = sanitize_input(user_input)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sanitized_input}
    ]
```

3. Meta-Routing (Intent-Based Secure Execution):
Classify user query and route to appropriate safe workflow.
Prevents misuse of sensitive pipelines
Reduces attack surface
Ensures controlled execution paths

```py
def route_query(msg):
    msg = sanitize_input(msg)

    if "bill" in msg:
        prompt = "Resolve billing issue:\n"
    elif "error" in msg:
        prompt = "Handle technical issue:\n"
    else:
        prompt = "Answer safely:\n"
    
    return call_llm(prompt + msg)
```

4. Dual LLM Validation (Output Verification):
Use multiple models to validate responses and reduce hallucinations.
Detects incorrect or inconsistent outputs
Adds redundancy and reliability
Useful in high-stakes systems

```py
def dual_llm_validate(prompt):
    response1 = call_llm(prompt, model="gpt-4")
    response2 = call_llm(prompt, model="mistral-7b")

    if response1.strip() == response2.strip():
        return response1
    
    arb_prompt = f"Which is correct?\nA: {response1}\nB: {response2}"
    return call_llm(arb_prompt)
```

5. Output Filtering & Data Protection:
Filter sensitive or unsafe information before returning output.
Prevents data leakage
Protects sensitive information
Ensures compliance and safe deployment

```py
def filter_output(response):
    sensitive_keywords = ["password", "api_key", "ssn"]

    for word in sensitive_keywords:
        if word in response.lower():
            return "Sensitive information detected. Response blocked."
    
    return response

```



* LLM Security: 
LLM security focuses on protecting models from misuse and malicious inputs.
It uses input validation, output filtering, and strict prompt control.
Techniques like least-privilege access and sandboxing limit system risks.
Monitoring and logging help detect anomalies and attack patterns.
Together, these ensure safe, reliable, and production-ready AI systems.


## Question B: Evaluating LLM Output Quality

Evaluating LLM output quality involves assessing both retrieval performance (For RAG) and answer generation quality using a combination of quatitative metrics and model-based evaluation.

#### Metrics
1. Retrieval Metrics: MRR, nDCG, Recall@k, Precision@k
2. Answer Generation Metrics: BELU, ROUGE, METEOR
3. Answers Quality Evaluation: LLM-as-a-judge, human evaluation


LLM-as-a-judge method used to measure the quality of answers:
LLM-as-a-judge method is used to score provided answers against critical like accuracy, completeness,
and relevance.

Steps to use LLM-as-a-judge method:
1. Prepare evaluation dataset: Create a dataset of questions and reference answers.
2. Design evaluation prompt: Create a prompt that instructs the LLM to evaluate the answers.

