# Module 2: Exploring Artificial Intelligence Use Cases and Applications

> **Course:** AWS Artificial Intelligence Practitioner (AIF-C01)
> **Source:** Exploring Artificial Intelligence Use Cases and Applications — AWS / Coursera
> **Renders best on GitHub** — Mermaid diagrams display automatically

---

## Table of Contents

1. [AI/ML/DL/GenAI — Quick Review](#1-aimldlgenai--quick-review)
2. [Real-World AI Use Cases by Industry](#2-real-world-ai-use-cases-by-industry)
3. [AI Applications](#3-ai-applications)
4. [When AI and ML Are (and Are Not) Appropriate](#4-when-ai-and-ml-are-and-are-not-appropriate)
5. [ML Techniques and Use Cases](#5-ml-techniques-and-use-cases)
6. [Generative AI Capabilities](#6-generative-ai-capabilities)
7. [Challenges of Generative AI](#7-challenges-of-generative-ai)
8. [Selecting a Generative AI Model](#8-selecting-a-generative-ai-model)
9. [Amazon Bedrock Model Reference](#9-amazon-bedrock-model-reference)
10. [Business Metrics for Generative AI](#10-business-metrics-for-generative-ai)
11. [Exam Cheat Sheet](#11-exam-cheat-sheet)

---

## 1. AI/ML/DL/GenAI — Quick Review

> Module 2 begins with a review of the same hierarchy from Module 1. Ensure these definitions are locked in.

```
╔══════════════════════════════════════════════════════════════╗
║  Artificial Intelligence (AI)                                ║
║  Intelligent systems mimicking human perception, reasoning,  ║
║  learning, problem-solving, and decision-making              ║
║  ╔══════════════════════════════════════════════════════╗    ║
║  ║  Machine Learning (ML)                               ║    ║
║  ║  Subset of AI — machines learn from data             ║    ║
║  ║  ╔══════════════════════════════════════════════╗    ║    ║
║  ║  ║  Deep Learning (DL)                          ║    ║    ║
║  ║  ║  Neurons and synapses — mimics the brain     ║    ║    ║
║  ║  ║  ╔══════════════════════════════════════╗    ║    ║    ║
║  ║  ║  ║  Generative AI                       ║    ║    ║    ║
║  ║  ║  ║  Subset of DL — creates new content  ║    ║    ║    ║
║  ║  ║  ║  without retraining or fine-tuning   ║    ║    ║    ║
║  ║  ║  ╚══════════════════════════════════════╝    ║    ║    ║
║  ║  ╚══════════════════════════════════════════════╝    ║    ║
║  ╚══════════════════════════════════════════════════════╝    ║
╚══════════════════════════════════════════════════════════════╝
```

### Generative AI — Key Capabilities (review)

Generative AI can create new content including:


| Content Type       | Examples                         |
| ------------------ | -------------------------------- |
| **Conversations**  | Chatbots, virtual assistants     |
| **Stories / Text** | Articles, summaries, reports     |
| **Images**         | Product photos, marketing assets |
| **Videos**         | Synthetic media, animations      |
| **Music**          | Compositions, sound design       |
| **Code**           | Functions, scripts, debugging    |


---

## 2. Real-World AI Use Cases by Industry

```mermaid
graph TD
    AI["🤖 AI Industry Applications"]

    AI --> ME["🎬 Media &\nEntertainment"]
    AI --> RT["🛍️ Retail"]
    AI --> HC["🏥 Healthcare"]
    AI --> LS["🔬 Life Sciences"]
    AI --> FS["💰 Financial\nServices"]
    AI --> MF["🏭 Manufacturing"]

    ME --> ME1["• Content generation\n  (scripts, dialogues, stories)\n• Virtual reality environments\n• Article / news generation"]
    RT --> RT1["• Product review summaries\n• Pricing optimization\n• Virtual try-ons\n• Store layout optimization"]
    HC --> HC1["• AWS HealthScribe\n  (clinical notes)\n• Personalized medicine\n• Medical imaging enhancement"]
    LS --> LS1["• Drug discovery\n• Protein folding prediction\n• Synthetic biology"]
    FS --> FS1["• Fraud detection\n• Portfolio management\n• Debt collection strategies"]
    MF --> MF1["• Predictive maintenance\n• Process optimization\n• Product design\n• Material science"]

    style AI fill:#2c3e50,color:#fff
    style ME fill:#1a4a7a,color:#fff
    style RT fill:#0F6E56,color:#fff
    style HC fill:#4a3a9a,color:#fff
    style LS fill:#7a4a1a,color:#fff
    style FS fill:#7a2a1a,color:#fff
    style MF fill:#1a4a4a,color:#fff
```



### Industry Use Cases — Expanded Reference

#### 🎬 Media and Entertainment


| Use Case           | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| Content generation | AI creates scripts, dialogues, and complete stories for films, TV, and games |
| Virtual reality    | AI builds immersive, interactive virtual environments                        |
| News generation    | AI generates articles and summaries from raw data or events                  |


#### 🛍️ Retail


| Use Case                  | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| Product review summaries  | Condenses reviews so shoppers find relevant info quickly       |
| Pricing optimization      | Models scenarios to find optimal pricing for maximum profit    |
| Virtual try-ons           | Generates virtual customer models for online clothing shopping |
| Store layout optimization | Determines most efficient layouts to boost sales               |


#### 🏥 Healthcare


| Use Case              | Description                                                        |
| --------------------- | ------------------------------------------------------------------ |
| AWS HealthScribe      | Auto-generates clinical notes from patient-clinician conversations |
| Personalized medicine | Generates treatment plans based on patient genetics and lifestyle  |
| Medical imaging       | Enhances, reconstructs, or generates X-rays, MRIs, CT scans        |


#### 🔬 Life Sciences


| Use Case                   | Description                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| Drug discovery             | Generates molecular structures, accelerates drug development      |
| Protein folding prediction | Predicts 3D protein structures from amino acid sequences          |
| Synthetic biology          | Generates designs for engineered organisms or biological circuits |


#### 💰 Financial Services


| Use Case             | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| Fraud detection      | Creates synthetic datasets to train better fraud detection models |
| Portfolio management | Simulates market scenarios for investment portfolio creation      |
| Debt collection      | Generates optimal communication strategies for collections        |


#### 🏭 Manufacturing


| Use Case               | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| Predictive maintenance | Predicts maintenance schedules from historical production data |
| Process optimization   | Models production scenarios to optimize cost, time, resources  |
| Product design         | Generates multiple design options optimized for set parameters |
| Material science       | Generates new material compositions with desired properties    |


---

## 3. AI Applications

> These four core AI applications appear across all industries. Know which industry maps to which application.

```mermaid
graph LR
    subgraph CV["👁️ Computer Vision"]
        CV1["Interprets digital\nimages and videos"]
        CV2["Uses deep learning:\nimage classification,\nobject detection,\nimage segmentation"]
    end

    subgraph NLP["💬 Natural Language\nProcessing (NLP)"]
        NLP1["Interaction between\ncomputers and\nhuman language"]
        NLP2["Tasks: text classification,\nsentiment analysis,\nmachine translation,\nlanguage generation"]
    end

    subgraph IDP["📄 Intelligent Document\nProcessing (IDP)"]
        IDP1["Extracts and classifies\ninformation from\nunstructured data"]
        IDP2["Generates summaries\nand actionable insights\nUses OCR + NLP"]
    end

    subgraph FD["🚨 Fraud Detection"]
        FD1["Identifies and prevents\nfraudulent activities\nor unauthorized behavior"]
        FD2["Used in: financial\nservices, retail,\ntelecommunications"]
    end

    style CV fill:#1a4a7a,color:#fff
    style NLP fill:#0F6E56,color:#fff
    style IDP fill:#4a3a9a,color:#fff
    style FD fill:#7a2a1a,color:#fff
```



### Application → Industry Mapping

```
┌──────────────────────────┬──────────────────────────────────────────────────────┐
│ APPLICATION              │ INDUSTRY EXAMPLES                                    │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│ Computer Vision          │ • Automotive: self-driving cars (safety)             │
│                          │ • Healthcare: medical imaging (better diagnosis)     │
│                          │ • Public safety: facial recognition (crime deterrent)│
│ Business value           │ Enhance customer experience, Improve operations      │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│ NLP                      │ • Insurance: extract policy numbers, redact PII      │
│                          │ • Telecom: personalized recommendations from texts   │
│                          │ • Education: Q&A chatbots for students               │
│ Business value           │ Sensitive data redaction, Customer engagement        │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│ IDP                      │ • Financial services: mortgage application processing│
│                          │ • Legal: process contracts, agreements, court filings│
│                          │ • Healthcare: claims processing, doctor's notes      │
│ Business value           │ Improve business operations, Automation              │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│ Fraud Detection          │ • Financial services: identity verification, AML     │
│                          │ • Retail: protect from financial losses, account data│
│                          │ • Telecom: roaming fraud, subscription fraud         │
│ Business value           │ Improve business operations                          │
└──────────────────────────┴──────────────────────────────────────────────────────┘
```

### IDP Components

```mermaid
graph LR
    RAW["📄 Raw Unstructured\nDocuments\n(invoices, contracts,\nforms, claims)"]
    OCR["🔍 OCR\nOptical Character\nRecognition\n(read the text)"]
    NLP2["💬 NLP\nUnderstand context,\nextract meaning"]
    ML2["⚙️ ML Models\nClassify, extract\nfields, validate"]
    OUT["✅ Structured Output\nExtracted data,\nsummaries,\nactionable insights"]

    RAW --> OCR --> NLP2 --> ML2 --> OUT

    style RAW fill:#2c3e50,color:#fff
    style OCR fill:#1a4a7a,color:#fff
    style NLP2 fill:#0F6E56,color:#fff
    style ML2 fill:#4a3a9a,color:#fff
    style OUT fill:#7a2a1a,color:#fff
```



---

## 4. When AI and ML Are (and Are Not) Appropriate

### Use AI/ML When:

```mermaid
graph TD
    Q1{"Can you solve this\nwith simple rules?"}
    Q1 -- "✅ Yes — rules are simple,\nclear, and limited" --> NO["❌ Don't use ML\nUse rule-based logic\nor simple computation"]
    Q1 -- "❌ No — too many variables,\noverlapping rules" --> Q2{"Is the problem\nat large scale?"}
    Q2 -- "Small scale\n(hundreds of records)" --> MAYBE["⚠️ ML may be overkill\nConsider simpler approaches"]
    Q2 -- "Large scale\n(millions of records)" --> YES["✅ Use ML\nML scales well;\nhumans cannot"]

    style NO fill:#7a2a1a,color:#fff
    style YES fill:#0F6E56,color:#fff
    style MAYBE fill:#7a4a1a,color:#fff
```



### Decision Table


| Situation                                      | Recommended Approach                    |
| ---------------------------------------------- | --------------------------------------- |
| Rules are simple, clear, and finite            | ❌ Don't use ML — use rule-based code    |
| Target value can be calculated by formula      | ❌ Don't use ML — use direct computation |
| Rules rely on many overlapping factors         | ✅ Use ML                                |
| Task needs to scale to millions of records     | ✅ Use ML                                |
| Human labeling is feasible and data is limited | Consider rule-based first               |
| Patterns are unknown and need to be discovered | ✅ Use unsupervised ML                   |


### Classic Example — Spam Filtering

```
Rule-based approach:          ML approach:
─────────────────────         ─────────────────────
IF subject contains           Trained on millions of
"FREE MONEY" → spam           labeled emails, learns
IF sender unknown → spam      complex patterns of
...hundreds more rules...     spam automatically.
Hard to maintain, fails       Scales to millions of
on new spam patterns          emails, improves over time
```

---

## 5. ML Techniques and Use Cases

### Full Taxonomy

```mermaid
graph TD
    ML["⚙️ Machine Learning\nTechniques"]

    ML --> SL["📌 Supervised Learning\nLabeled training data\nSupervisor = labels"]
    ML --> UL["🔍 Unsupervised Learning\nUnlabeled data\nDiscover hidden patterns"]
    ML --> RL["🏆 Reinforcement Learning\nRewards and penalties\nTrial and error"]

    SL --> CL["Classification\nAssign labels/categories\nto new data instances"]
    SL --> RG["Regression\nPredict continuous\nor numerical values"]

    UL --> CLS["Clustering\nGroup data by\nsimilar features"]
    UL --> DR["Dimensionality\nReduction\nReduce features while\npreserving information"]

    CL --> CL_EX["Use cases:\n• Fraud detection\n• Image classification\n• Customer retention\n• Medical diagnostics"]
    RG --> RG_EX["Use cases:\n• House price prediction\n• Weather forecasting\n• Market forecasting\n• Life expectancy\n• Population growth"]
    CLS --> CLS_EX["Use cases:\n• Customer segmentation\n• Targeted marketing\n• Recommendation systems"]
    DR --> DR_EX["Use cases:\n• Big data visualization\n• Meaningful compression\n• Structure discovery\n• Feature elicitation"]
    RL --> RL_EX["Use cases:\n• AWS DeepRacer\n• Self-driving cars\n• Game playing\n• Robot navigation"]

    style ML fill:#2c3e50,color:#fff
    style SL fill:#0F6E56,color:#fff
    style UL fill:#1a4a7a,color:#fff
    style RL fill:#4a3a9a,color:#fff
    style CL fill:#0a5e46,color:#fff
    style RG fill:#0a5e46,color:#fff
    style CLS fill:#1a3a6a,color:#fff
    style DR fill:#1a3a6a,color:#fff
```



### Supervised Learning — Deep Dive

```
HOW IT WORKS:
  Labeled data ──→ Model trains ──→ Learns patterns ──→ Predicts on new data

EXAMPLE (Image Classification):
  [Car photo #1] + label "car" ─┐
  [Car photo #2] + label "car" ─┼──→ [Model] ──→ New unlabeled photo ──→ "car"
  [Car photo #3] + label "car" ─┘

CLASSIFICATION vs REGRESSION:
  ┌─────────────────┬────────────────────────────────────────┐
  │ Classification  │ Output is a CATEGORY / LABEL           │
  │                 │ Example: Is this email spam? Yes / No  │
  │                 │ Example: Is this tumor malignant?      │
  ├─────────────────┼────────────────────────────────────────┤
  │ Regression      │ Output is a NUMBER / CONTINUOUS VALUE  │
  │                 │ Example: What will the house sell for? │
  │                 │ Example: What will tomorrow's temp be? │
  └─────────────────┴────────────────────────────────────────┘
```

### Unsupervised Learning — Deep Dive

```
HOW IT WORKS:
  Unlabeled data ──→ Model finds structure ──→ Creates its own labels/groups

EXAMPLE (Clustering):
  Customer transactions (no labels) ──→ Model discovers:
  Group A: High-spend, frequent buyers   (→ "Premium customers")
  Group B: Low-spend, occasional buyers  (→ "Casual shoppers")
  Group C: Bulk orders, low frequency    (→ "Business buyers")

CLUSTERING vs DIMENSIONALITY REDUCTION:
  ┌──────────────────────────┬──────────────────────────────────────────┐
  │ Clustering               │ Groups data points by similarity         │
  │                          │ Example: segment customers by behavior   │
  ├──────────────────────────┼──────────────────────────────────────────┤
  │ Dimensionality Reduction │ Reduces number of features in a dataset  │
  │                          │ Keeps most important information         │
  │                          │ Example: compress 100 features → 10      │
  └──────────────────────────┴──────────────────────────────────────────┘
```

### Reinforcement Learning — AWS DeepRacer Example

```mermaid
sequenceDiagram
    participant E as Environment (Racetrack)
    participant A as Agent (Virtual Car)
    participant R as Reward Function

    A->>E: Takes action (steering + throttle)
    E->>A: Returns new state (position on track)
    E->>R: Evaluates action
    R->>A: Reward (+) if on track, (-) if off track
    Note over A: Updates policy to\nmaximize future rewards
    A->>E: Takes next action (improved)
    Note over A,E: Repeats thousands of times\nuntil car learns to race
```



---

## 6. Generative AI Capabilities

```mermaid
graph TD
    GEN["✨ Generative AI\nCapabilities"]

    GEN --> A["🔄 Adaptability\nAdapts to various tasks\nand domains from data"]
    GEN --> R["⚡ Responsiveness\nGenerates content in\nreal-time for chatbots\nand virtual assistants"]
    GEN --> S["🎯 Simplicity\nAutomates complex\ncontent creation tasks"]
    GEN --> C["🎨 Creativity\nGenerates novel ideas,\ndesigns, and solutions\nby recombining elements"]
    GEN --> D["📊 Data Efficiency\nCan learn from small\namounts of data and\nstill generate new samples"]
    GEN --> P["👤 Personalization\nCreates personalized\ncontent for individual\npreferences"]
    GEN --> SC["📈 Scalability\nGenerates large amounts\nof content quickly\nafter training"]

    style GEN fill:#2c3e50,color:#fff
    style A fill:#1a4a7a,color:#fff
    style R fill:#0F6E56,color:#fff
    style S fill:#4a3a9a,color:#fff
    style C fill:#7a4a1a,color:#fff
    style D fill:#7a2a1a,color:#fff
    style P fill:#1a4a4a,color:#fff
    style SC fill:#4a1a7a,color:#fff
```



### Capabilities Summary Table


| Capability          | Definition                        | Business Example                                  |
| ------------------- | --------------------------------- | ------------------------------------------------- |
| **Adaptability**    | Flexible across tasks and domains | Same FM used for HR, legal, and marketing content |
| **Responsiveness**  | Generates content in real-time    | Live chatbot responses to customer queries        |
| **Simplicity**      | Automates content creation        | Auto-drafts product descriptions from specs       |
| **Creativity**      | Generates novel ideas             | New product design variations from constraints    |
| **Data Efficiency** | Learns from small datasets        | Fine-tuned on 500 examples; generalizes well      |
| **Personalization** | Tailors content to individuals    | Personalized email campaigns per customer         |
| **Scalability**     | Produces content at massive scale | Generate 1M product descriptions overnight        |


> **⚠️ Exam Key:** The exam may ask you to select capabilities from a list. The core 7 are: Adaptability, Responsiveness, Simplicity, Creativity, Data Efficiency, Personalization, Scalability. **Conversion rate and ARPU are business metrics — NOT capabilities.**

---

## 7. Challenges of Generative AI

```mermaid
graph TD
    CH["⚠️ Generative AI\nChallenges"]

    CH --> RV["📋 Regulatory\nViolations\nMay expose PII\nor violate laws"]
    CH --> SR["👥 Social Risks\nUnwanted content\nreflecting negatively\non organization"]
    CH --> PR["🔒 Data Security\n& Privacy\nPersonal data shared\nwith model"]
    CH --> TX["☠️ Toxicity\nInflammatory,\noffensive, or\ninappropriate content"]
    CH --> HA["👻 Hallucinations\nInaccurate responses\nnot grounded in\ntraining data"]
    CH --> IN["🔍 Interpretability\nUsers misinterpret\nmodel output →\nwrong decisions"]
    CH --> ND["🎲 Nondeterminism\nDifferent outputs\nfor same input →\nunreliable results"]

    style CH fill:#2c3e50,color:#fff
    style RV fill:#7a2a1a,color:#fff
    style SR fill:#7a2a1a,color:#fff
    style PR fill:#7a2a1a,color:#fff
    style TX fill:#7a2a1a,color:#fff
    style HA fill:#7a2a1a,color:#fff
    style IN fill:#7a2a1a,color:#fff
    style ND fill:#7a2a1a,color:#fff
```



### Challenges — Risk and Mitigation

```
┌─────────────────────┬──────────────────────────────┬─────────────────────────────────────┐
│ CHALLENGE           │ RISK                         │ MITIGATION                          │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Regulatory          │ Generates output that        │ Data anonymization; audits and       │
│ Violations          │ exposes PII or violates       │ assessments of training data         │
│                     │ compliance regulations        │                                     │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Social Risks        │ Unwanted content reflects    │ Test and evaluate all models         │
│                     │ negatively on organization   │ before production deployment         │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Data Security       │ Personal information shared  │ Encrypt data in transit and at rest; │
│ & Privacy           │ with model may violate       │ IAM, KMS, VPC controls;              │
│                     │ privacy laws                 │ Amazon Bedrock Guardrails            │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Toxicity            │ Model generates offensive    │ Curate training data; remove toxic   │
│                     │ or inappropriate content     │ phrases; use guardrail models        │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Hallucinations      │ Model generates inaccurate   │ Teach users to verify all outputs;   │
│                     │ responses not based in       │ mark generated content as unverified;│
│                     │ reality                      │ use RAG for grounded responses       │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Interpretability    │ Users misinterpret output,   │ Use domain knowledge for model dev;  │
│                     │ leading to wrong decisions   │ provide key info in inputs           │
├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
│ Nondeterminism      │ Different outputs for same   │ Test model repeatedly; compare       │
│                     │ input — unreliable results   │ outputs; identify sources of drift   │
└─────────────────────┴──────────────────────────────┴─────────────────────────────────────┘
```

### Hallucinations — Explained

```
What happens:
  User asks: "What is the capital of Australia?"
  Model says: "The capital of Australia is Sydney." ← HALLUCINATION
                                                       (Correct answer: Canberra)

Why it happens:
  The model generates statistically likely text — not verified facts.
  It "sounds confident" but is not checking against a source of truth.

Mitigation options:
  1. RAG — ground responses in retrieved documents
  2. Teach users to always verify AI-generated content
  3. Mark all generated content as "unverified"
  4. Amazon Bedrock Guardrails — filter outputs
```

---

## 8. Selecting a Generative AI Model

### Five Key Factors

```mermaid
graph TD
    SEL["🎯 Selecting a\nGenerative AI Model"]

    SEL --> MT["📦 Model Types\nChoose model optimized\nfor your specific task\n(text, image, code, multimodal)"]
    SEL --> PR["📊 Performance\nRequirements\nAccuracy, reliability,\nconsistency across datasets"]
    SEL --> CA["⚡ Capabilities\nText generation? Image?\nCode? Multimodal?\nDegree of customization?"]
    SEL --> CO["⚙️ Constraints\nGPU/CPU/memory\nData availability\nOn-premises vs cloud"]
    SEL --> CM["✅ Compliance\nFairness, transparency,\naccountability, toxicity,\nhallucination, regulations"]

    SEL --> CST["💰 Cost\nLarger models = more precise\nbut expensive + few deploy options\nSmaller = cheaper + faster + flexible"]

    style SEL fill:#2c3e50,color:#fff
    style MT fill:#1a4a7a,color:#fff
    style PR fill:#0F6E56,color:#fff
    style CA fill:#4a3a9a,color:#fff
    style CO fill:#7a4a1a,color:#fff
    style CM fill:#7a2a1a,color:#fff
    style CST fill:#1a4a4a,color:#fff
```



### Cost Trade-off: Model Size

```
LARGER MODELS:                    SMALLER MODELS:
──────────────                    ───────────────
✅ More precise / accurate        ✅ Cheaper to run
✅ Better at complex tasks        ✅ Faster inference
❌ More expensive                 ✅ More deployment options
❌ Fewer deployment options       ❌ Less accurate on complex tasks
❌ Higher latency                 ❌ May need fine-tuning for quality
```

### Compliance Factors to Consider


| Factor                          | Definition                                              |
| ------------------------------- | ------------------------------------------------------- |
| **Fairness**                    | Does the model treat all groups equitably?              |
| **Transparency / Traceability** | Can you explain how the model reached its output?       |
| **Accountability**              | Is there a clear owner responsible for model decisions? |
| **Hallucination**               | Does the model produce false or unverified content?     |
| **Toxicity**                    | Does the model generate offensive or harmful content?   |


---

## 9. Amazon Bedrock Model Reference

```mermaid
graph LR
    subgraph Bedrock["Amazon Bedrock — Available Foundation Models"]
        AI21["AI21 Labs\nJurassic-2\n\nTasks: Text generation,\nsummarization,\nparaphrasing, chat,\ninformation extraction\n\nUse case: Financial services\n(summarize documents)"]

        AMZ["Amazon\nTitan\n\nTasks: Text summarization,\nclassification, Q&A,\nextraction, embeddings,\nsearch\n\nUse case: Customer service\n(real-time summaries)"]

        ANT["Anthropic\nClaude\n\nTasks: Content generation,\ntranslation, Q&A,\nsummarization, code\nexplanation & generation\n\nUse case: Legal (parse docs);\nDeveloper (code/debug)"]

        STB["Stability AI\nStable Diffusion\n\nTasks: Generate\nphotorealistic images\nfrom text, improve\nimage quality\n\nUse case: Gaming\n(characters, scenes);\nAdvertising (campaigns)"]

        COH["Cohere\nCommand\n\nTasks: Text generation,\nextraction, Q&A,\nsummarization\n\nUse case: Customer service\n(chatbots); Healthcare\n(summarize long text)"]

        META["Meta\nLlama\n\nTasks: Q&A, chat,\nsummarization,\nparaphrasing, sentiment,\ntext generation\n\nUse case: Customer service\nsupport (chatbots)"]
    end

    style AI21 fill:#1a4a7a,color:#fff
    style AMZ fill:#7a4a1a,color:#fff
    style ANT fill:#0F6E56,color:#fff
    style STB fill:#4a3a9a,color:#fff
    style COH fill:#7a2a1a,color:#fff
    style META fill:#1a4a4a,color:#fff
```



### Model Quick-Reference Table


| Provider         | Model            | Best Tasks                                    | Exam Use Case Trigger                           |
| ---------------- | ---------------- | --------------------------------------------- | ----------------------------------------------- |
| **AI21 Labs**    | Jurassic-2       | Text generation, summarization, paraphrasing  | "Summarize lengthy financial documents"         |
| **Amazon**       | Titan            | Summarization, Q&A, embeddings, search        | "Customer service summaries"; "semantic search" |
| **Anthropic**    | Claude           | Code generation, legal docs, Q&A, translation | "Code debugging"; "parse legal documents"       |
| **Stability AI** | Stable Diffusion | Text-to-image generation                      | "Generate product images"; "advertising assets" |
| **Cohere**       | Command          | Text generation, summarization, Q&A           | "Support chatbots"; "healthcare text summaries" |
| **Meta**         | Llama            | Chat, Q&A, sentiment analysis                 | "Customer service chatbots"                     |


---

## 10. Business Metrics for Generative AI

```mermaid
graph TD
    BM["📊 Business Metrics\nfor Generative AI"]

    BM --> US["😊 User Satisfaction\nGather user feedback\non AI-generated content\nand recommendations"]
    BM --> AR["💰 Average Revenue\nPer User (ARPU)\nAvg revenue generated\nper user attributed to\nthe AI application"]
    BM --> CD["🌐 Cross-Domain\nPerformance\nModel's ability to\nperform across different\ndomains or industries"]
    BM --> CR["🔄 Conversion Rate\nMonitor % of users who\ncomplete desired action\n(purchase, sign-up, etc.)"]
    BM --> EF["⚡ Efficiency\nEvaluates resource\nutilization, computation\ntime, and scalability"]

    style BM fill:#2c3e50,color:#fff
    style US fill:#0F6E56,color:#fff
    style AR fill:#1a4a7a,color:#fff
    style CD fill:#4a3a9a,color:#fff
    style CR fill:#7a4a1a,color:#fff
    style EF fill:#7a2a1a,color:#fff
```



### Business Metrics — Detailed Reference

```
┌──────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ METRIC                   │ WHAT IT MEASURES                │ EXAMPLE USE CASE                │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ User Satisfaction        │ How satisfied users are with    │ E-commerce: surveys after AI    │
│                          │ AI-generated outputs            │ product recommendations         │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ ARPU                     │ Average revenue per user from   │ E-commerce: revenue from users  │
│ (Avg Revenue Per User)   │ AI-driven interactions          │ who engaged with AI suggestions │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Cross-Domain Performance │ Model effectiveness across      │ Multi-category e-commerce:      │
│                          │ multiple domains/industries     │ performance across all product  │
│                          │                                 │ categories and regions          │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Conversion Rate          │ % of users who complete a       │ Online store: % of visitors who │
│                          │ desired action (purchase, etc.) │ buy after AI recommendations    │
├──────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ Efficiency               │ Resource usage, compute time,   │ Manufacturing: AI optimizes     │
│                          │ and scalability                 │ production line resource usage  │
└──────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

> **⚠️ Exam Key:** Business metrics ≠ GenAI capabilities. The exam may try to mix them. Conversion Rate, ARPU, and Cross-Domain Performance are **metrics**, not capabilities. Personalization, Scalability, and Simplicity are **capabilities**, not metrics.

---

## 11. Exam Cheat Sheet

### Industry → AI Use Case Mapping

```
Autonomous driving / self-driving cars     → Computer Vision
Medical imaging (X-rays, MRIs, CT scans)   → Computer Vision / Healthcare AI
Clinical notes from conversations          → AWS HealthScribe (NLP)
Drug discovery / protein folding           → Life Sciences AI
Fraud detection / AML                      → Fraud Detection / Financial AI
Portfolio management / market simulation   → Financial Services AI
Predictive maintenance on machines         → Manufacturing AI
Product descriptions at scale              → Retail AI / Generative AI
Virtual try-ons for online shoppers        → Retail AI
Article / news generation                  → Media & Entertainment AI
Extract fields from contracts/invoices     → IDP (Intelligent Document Processing)
Insurance: extract policy numbers, PII     → NLP
Student Q&A chatbots in education          → NLP
Telecom: personalized SMS recommendations  → NLP
```

### ML Technique → Task Mapping

```
Predicting a NUMBER (house price, temperature)     → Regression (Supervised)
Predicting a CATEGORY (spam/not spam, fraud/not)   → Classification (Supervised)
Grouping similar items (customer segments)         → Clustering (Unsupervised)
Reducing number of features in a dataset           → Dimensionality Reduction (Unsupervised)
Agent learns through rewards/penalties             → Reinforcement Learning
AWS DeepRacer                                      → Reinforcement Learning
```

### GenAI Capabilities vs. Business Metrics

```
CAPABILITIES (what the model can do):          BUSINESS METRICS (how we measure success):
• Adaptability                                 • User Satisfaction
• Responsiveness                               • Average Revenue Per User (ARPU)
• Simplicity                                   • Cross-Domain Performance
• Creativity                                   • Conversion Rate
• Data Efficiency                              • Efficiency
• Personalization
• Scalability
```

### GenAI Challenges — One-Line Summary

```
Regulatory violations  → Outputs may expose PII or violate compliance rules
Social risks           → Content may damage organizational reputation
Data security/privacy  → Sensitive data shared with model may leak
Toxicity               → Model may generate offensive or harmful content
Hallucinations         → Model confidently states things that are false
Interpretability       → Users misunderstand model outputs → bad decisions
Nondeterminism         → Same input produces different outputs → unreliable
```

### Model Selection Factors

```
1. Model type         → Does it match your task? (text, image, code, multimodal)
2. Performance        → Accuracy, reliability, consistency
3. Capabilities       → Does it support your specific requirements?
4. Constraints        → GPU, memory, on-prem vs cloud
5. Compliance         → Fairness, transparency, toxicity, hallucinations
6. Cost               → Larger = precise but expensive; smaller = fast and flexible
```

---

*Last updated: 2026 | AWS AI Practitioner (AIF-C01) | Module 2*