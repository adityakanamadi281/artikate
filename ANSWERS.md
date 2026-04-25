# Section 1: Diagnose a Failing LLM Pipeline

## Problem 1: Hallucinated Pricing

### Investigation
I first checked whether the model’s pricing answers were grounded in retrieved, up-to-date data or coming from its internal memory. I compared incorrect responses with the actual pricing source of truth.

### Ruled Out
- Temperature randomness: Re-running queries at low temperature still produced the same incorrect prices.
- Model capability issue: The model can answer correctly when given proper context.

### Root Cause
This is a retrieval issue.

The model generated answers without access to correct or fresh pricing data. Since pricing is dynamic, relying on model memory leads to stale or hallucinated responses.

### How to Distinguish Causes
- Prompt issue → Check if prompt enforces use of retrieved data  
- Retrieval issue → Verify if correct pricing appears in retrieved context  
- Temperature issue → Test consistency at temperature=0  
- Knowledge cutoff → Disable retrieval and observe outdated answers  

### Concrete Fix
- Use a structured pricing database  
- Enforce retrieval validation before answering  
- Add freshness metadata and cache invalidation  
- Require evidence-backed answers or refusal  

---

## Problem 2: Language Switching

### Investigation
I analyzed the instruction hierarchy (system prompt + user message). The system prompt is written in English, while users interact in Hindi or Arabic.

### Ruled Out
- Model limitation: The model supports multilingual responses  
- Randomness: The issue is consistent under certain conditions  

### Root Cause
The system prompt does not enforce language matching, so the model defaults to English.

### Mechanism
- System prompt has higher priority  
- Without language constraint, model follows English instructions  

### Concrete Fix

```
System:
You are a multilingual customer support assistant.

Always respond in the same language as the user's latest message.
Do not default to English unless the user writes in English.

If the message contains multiple languages, use the dominant language.
Preserve product names, URLs, and proper nouns exactly as written.

If unsure about the language, ask a clarification question in the user's last-used language.
```

### Why This Works
- Explicit system-level rule  
- Based on latest user message  
- Works for all languages  

---

## Problem 3: Latency Degradation

### Investigation
I focused on issues that worsen over time without code changes.

### Possible Causes
1. Traffic growth → queueing / worker saturation  
2. Retrieval slowdown → larger index  
3. External API delays  

### Investigation Priority
1. Queue wait time  
2. Retrieval latency  
3. Model/API latency  

### Root Cause
Capacity pressure + retrieval growth.

### Concrete Fix
- Add tracing (queue, retrieval, prompt, model)  
- Limit context size  
- Optimize retrieval index  
- Scale concurrency  
- Add caching  

---

## Post-Mortem Summary

The chatbot issues were caused by three system weaknesses.

Pricing errors occurred because the bot did not reliably check a live pricing source and instead used outdated information. Language issues happened because the system instructions did not enforce matching the user’s language, causing fallback to English. Latency increased due to higher usage, leading to delays in processing and retrieval.

The fixes include grounding pricing answers in verified data, enforcing strict language rules, and improving system performance through monitoring and scaling. These changes will make the system more accurate, consistent, and faster.

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
###  Prompt Ingestion:
Prompt ingestion is the structured process of handling user inputs in LLM systems.
It includes input collection, sanitization, validation, and intent classification.
Inputs are treated as untrusted and cleaned to prevent prompt injection attacks.
The system routes queries to appropriate workflows using controlled templates.
This ensures accuracy, scalability, and safe execution of LLM pipelines.

#### 1. Input Sanitization:
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

#### 2. Secure Prompt Construction (Instruction Hierarchy):
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

#### 3. Meta-Routing (Intent-Based Secure Execution):
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

#### 4. Dual LLM Validation (Output Verification):
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

#### 5. Output Filtering & Data Protection:
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



### LLM Security: 
* LLM security focuses on protecting models from misuse and malicious inputs.
* It uses input validation, output filtering, and strict prompt control.
* Techniques like least-privilege access and sandboxing limit system risks.
* Monitoring and logging help detect anomalies and attack patterns.
* Together, these ensure safe, reliable, and production-ready AI systems.


## Question B: Evaluating LLM Output Quality

Evaluating LLM output quality involves assessing both retrieval performance (For RAG) and answer generation quality using a combination of quatitative metrics and model-based evaluation.

#### Metrics
1. Retrieval Metrics: MRR, nDCG, Recall@k, Precision@k
2. Answer Generation Metrics: BELU, ROUGE, METEOR
3. Answers Quality Evaluation: LLM-as-a-judge, human evaluation, Faithfulness, Toxicity


LLM-as-a-judge method used to measure the quality of answers:
LLM-as-a-judge method is used to score provided answers against critical like accuracy, completeness,
and relevance.

Steps to use LLM-as-a-judge method:
1. Prepare evaluation dataset: Create a dataset of questions and reference answers.
2. Design evaluation prompt: Create a prompt that instructs the LLM to evaluate the answers.

