# Module 1: Fundamentals of Machine Learning and Artificial Intelligence

> **Course:** AWS Artificial Intelligence Practitioner (AIF-C01)
> **Source:** Fundamentals of Machine Learning and Artificial Intelligence — AWS / Coursera
> **Renders best on GitHub** — Mermaid diagrams display automatically

---

## Table of Contents

1. [The AI/ML Hierarchy](#1-the-aiml-hierarchy)
2. [Training Data](#2-training-data)
3. [The Machine Learning Process](#3-the-machine-learning-process)
4. [ML Learning Types](#4-ml-learning-types)
5. [Inferencing](#5-inferencing)
6. [Deep Learning and Neural Networks](#6-deep-learning-and-neural-networks)
7. [Computer Vision and NLP](#7-computer-vision-and-nlp)
8. [Foundation Models](#8-foundation-models)
9. [Large Language Models](#9-large-language-models)
10. [Generative Model Types](#10-generative-model-types)
11. [FM Optimization Techniques](#11-fm-optimization-techniques)
12. [AWS AI/ML Services Stack](#12-aws-aiml-services-stack)
13. [Cost Considerations](#13-cost-considerations)
14. [Exam Cheat Sheet](#14-exam-cheat-sheet)

---

## 1. The AI/ML Hierarchy

> AI is the broadest concept. Each inner layer is a more specific approach to achieving AI.

```mermaid
graph TD
    AI["🤖 Artificial Intelligence (AI)\nBroadest field — any system mimicking human intelligence\nExamples: expert systems, robotics, planning"]
    ML["📊 Machine Learning (ML)\nMachines learn from data to improve performance\nExamples: decision trees, random forests, SVMs"]
    DL["🧠 Deep Learning (DL)\nMulti-layered neural networks inspired by the brain\nExamples: CNNs, RNNs, Amazon Rekognition"]
    GenAI["✨ Generative AI\nCreates new content using pre-trained models\nExamples: LLMs, Diffusion models, Amazon Bedrock"]

    AI --> ML
    ML --> DL
    DL --> GenAI

    style AI fill:#1a4a7a,color:#fff,stroke:#378ADD
    style ML fill:#0F6E56,color:#fff,stroke:#2ecc71
    style DL fill:#4a3a9a,color:#fff,stroke:#9b59b6
    style GenAI fill:#7a2a1a,color:#fff,stroke:#e74c3c
```



### Visual: Nested Rings

```
╔══════════════════════════════════════════════════════╗
║          Artificial Intelligence (AI)                ║
║   (Expert systems, robotics, planning, ML, ...)      ║
║  ╔════════════════════════════════════════════╗      ║
║  ║         Machine Learning (ML)              ║      ║
║  ║   (Decision trees, SVMs, clustering ...)   ║      ║
║  ║  ╔════════════════════════════════════╗    ║      ║
║  ║  ║       Deep Learning (DL)           ║    ║      ║
║  ║  ║  (Neural networks, CNNs, RNNs ...) ║    ║      ║
║  ║  ║  ╔══════════════════════════════╗  ║    ║      ║
║  ║  ║  ║      Generative AI           ║  ║    ║      ║
║  ║  ║  ║  (LLMs, Diffusion, GANs ...) ║  ║    ║      ║
║  ║  ║  ╚══════════════════════════════╝  ║    ║      ║
║  ║  ╚════════════════════════════════════╝    ║      ║
║  ╚════════════════════════════════════════════╝      ║
╚══════════════════════════════════════════════════════╝
```

### Key Definitions


| Concept           | Definition                                | AWS Example                    |
| ----------------- | ----------------------------------------- | ------------------------------ |
| **AI**            | Any system mimicking human intelligence   | Amazon Lex (conversational AI) |
| **ML**            | Machines learn patterns from data         | Amazon SageMaker AI            |
| **Deep Learning** | Multi-layered neural networks             | Amazon Rekognition             |
| **Generative AI** | Creates new content from learned patterns | Amazon Bedrock                 |


> **⚠️ Exam Key:** Generative AI adapts pre-trained deep learning models **without requiring full retraining** for every new task.

---

## 2. Training Data

> **"Garbage in, garbage out"** — your model is only as good as its training data. Data preparation is the single most critical stage of ML.

### Data Type Matrix

```
                    ┌──────────────────┬──────────────────────┐
                    │   STRUCTURED     │   UNSTRUCTURED       │
                    │ (rows & columns) │  (no fixed format)   │
  ┌─────────────────┼──────────────────┼──────────────────────┤
  │                 │                  │                      │
  │    LABELED      │ • CSV with tags  │ • Tagged images      │
  │  (has a target) │ • Database with  │ • Annotated emails   │
  │                 │   known outcomes │ • Transcribed audio  │
  │                 │                  │                      │
  │  → Supervised   │                  │                      │
  │    learning     │                  │                      │
  ├─────────────────┼──────────────────┼──────────────────────┤
  │                 │                  │                      │
  │   UNLABELED     │ • Raw databases  │ • Photos (no tags)   │
  │  (no target)    │ • Transactions   │ • Raw audio files    │
  │                 │   with no labels │ • Unannotated videos │
  │                 │                  │                      │
  │  → Unsupervised │                  │                      │
  │    learning     │                  │                      │
  └─────────────────┴──────────────────┴──────────────────────┘
```

### Structured vs. Unstructured

```mermaid
graph LR
    subgraph Structured["📋 Structured Data"]
        T1["Tabular\nSpreadsheets, CSV,\nDatabases, SQL tables"]
        T2["Time-Series\nStock prices,\nSensor readings,\nWeather data"]
    end
    subgraph Unstructured["📂 Unstructured Data"]
        U1["Text\nEmails, articles,\nSocial media posts,\nDocuments"]
        U2["Image / Video\nPhotos, video frames,\nMedical scans"]
        U3["Audio\nCall recordings,\nPodcasts, music"]
    end

    style Structured fill:#1a4a7a,color:#fff,stroke:#378ADD
    style Unstructured fill:#4a3a9a,color:#fff,stroke:#9b59b6
```



### Data Preparation Pipeline

```mermaid
flowchart LR
    A["📥 Collection\nGather raw data"] --> B["🧹 Cleaning\nFix errors,\nremove dupes,\nhandle nulls"]
    B --> C["🔄 Transformation\nNormalize,\nencode,\nscale"]
    C --> D["✂️ Splitting\nTrain / Validation\n/ Test sets"]
    D --> E["🏷️ Labeling\nAnnotate for\nsupervised tasks"]
    E --> F["🚀 Ready for\nTraining"]

    style A fill:#2c3e50,color:#fff
    style B fill:#1a4a7a,color:#fff
    style C fill:#0F6E56,color:#fff
    style D fill:#4a3a9a,color:#fff
    style E fill:#7a4a1a,color:#fff
    style F fill:#7a2a1a,color:#fff
```



> **⚠️ Exam Key:** When a model has poor accuracy, the most likely cause is **bad training data**, not the algorithm. Fix the data first.

---

## 3. The Machine Learning Process

```mermaid
flowchart TD
    A["1️⃣ Collect Data\nRaw, messy data\nfrom various sources"] --> B["2️⃣ Prepare Data\nClean, label,\ntransform, split"]
    B --> C["3️⃣ Choose Algorithm\nSupervised / Unsupervised /\nReinforcement"]
    C --> D["4️⃣ Train Model\nFeed training data\ninto algorithm"]
    D --> E["5️⃣ Evaluate\nTest on held-out data\ncheck metrics"]
    E --> F{"Performance\nacceptable?"}
    F -- "✅ Yes" --> G["6️⃣ Deploy\nAPI / endpoint\nor batch job"]
    F -- "❌ No" --> B
    G --> H["7️⃣ Monitor & Improve\nCollect feedback,\nretrain as needed"]
    H --> B

    style A fill:#2c3e50,color:#fff
    style B fill:#1a4a7a,color:#fff
    style C fill:#0F6E56,color:#fff
    style D fill:#4a3a9a,color:#fff
    style E fill:#7a4a1a,color:#fff
    style F fill:#555,color:#fff
    style G fill:#7a2a1a,color:#fff
    style H fill:#1a4a4a,color:#fff
```



---

## 4. ML Learning Types

```mermaid
graph TD
    MLT["Machine Learning Types"]

    MLT --> SL["📌 Supervised Learning\nUses LABELED data\nGoal: predict output for new inputs"]
    MLT --> UL["🔍 Unsupervised Learning\nUses UNLABELED data\nGoal: discover hidden patterns"]
    MLT --> RL["🏆 Reinforcement Learning\nUses REWARDS & PENALTIES\nGoal: maximize cumulative reward"]
    MLT --> SSL["🔀 Semi-Supervised\nUses SMALL labeled + LARGE unlabeled\nGoal: best of both worlds"]

    SL --> SLE["Examples:\n• Spam filter\n• Loan approval\n• Fraud detection\nAWS: SageMaker, Fraud Detector"]
    UL --> ULE["Examples:\n• Customer segmentation\n• Anomaly detection\n• Topic modeling\nAWS: SageMaker k-means"]
    RL --> RLE["Examples:\n• Self-driving cars\n• Game playing\n• Robot navigation\nAWS: AWS DeepRacer"]
    SSL --> SSLE["Examples:\n• Medical imaging\n• Document classification\nAWS: SageMaker Ground Truth"]

    style MLT fill:#2c3e50,color:#fff
    style SL fill:#0F6E56,color:#fff
    style UL fill:#1a4a7a,color:#fff
    style RL fill:#4a3a9a,color:#fff
    style SSL fill:#7a4a1a,color:#fff
```



### Side-by-Side Comparison

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   SUPERVISED        │   UNSUPERVISED      │   REINFORCEMENT     │
│   LEARNING          │   LEARNING          │   LEARNING          │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Uses labeled data   │ Uses unlabeled data │ No labels — uses    │
│                     │                     │ rewards/penalties   │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Tasks:              │ Tasks:              │ Tasks:              │
│ • Classification    │ • Clustering        │ • Game playing      │
│ • Regression        │ • Anomaly detection │ • Robotics          │
│                     │ • Dimensionality    │ • Ad bidding        │
│                     │   reduction         │                     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Example:            │ Example:            │ Example:            │
│ Spam filter —       │ Customer            │ AWS DeepRacer car   │
│ emails labeled      │ segmentation —      │ learns to race via  │
│ spam/not spam       │ group by behavior   │ reward signals      │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ AWS: SageMaker,     │ AWS: SageMaker      │ AWS: DeepRacer,     │
│ Fraud Detector      │ (k-means)           │ SageMaker RL        │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Quick Decision Guide

```
Has labeled data with known outcomes?         ──→  Supervised learning
No labels — find hidden groups / patterns?    ──→  Unsupervised learning
Agent optimizes through trial and error?      ──→  Reinforcement learning
Lots of data but only some is labeled?        ──→  Semi-supervised learning
```

---

## 5. Inferencing

> **Inferencing** = using a trained model to make predictions on new, unseen data.

### End-to-End ML Pipeline

```mermaid
flowchart LR
    A["📦 Collect\nData"] --> B["🧹 Prepare\nData"]
    B --> C["⚙️ Choose\nAlgorithm"]
    C --> D["🎓 Train\nModel"]
    D --> E["📊 Evaluate"]
    E --> F["🚀 Deploy"]
    F --> G["🔮 INFERENCE\nPredict on\nnew data"]

    G -.->|"Poor performance\nretrain"| B

    style G fill:#7a2a1a,color:#fff,stroke:#e74c3c,stroke-width:2px
```



### Batch vs. Real-Time

```mermaid
graph LR
    subgraph Batch["🗂️ Batch Inferencing"]
        B1["Large dataset\narrives"]
        B2["Process all\nat once"]
        B3["Results saved\nto S3/DB"]
        B1 --> B2 --> B3
    end

    subgraph Realtime["⚡ Real-Time Inferencing"]
        R1["Single input\narrives"]
        R2["Model responds\nin milliseconds"]
        R3["Immediate\nresponse returned"]
        R1 --> R2 --> R3
    end

    style Batch fill:#1a4a7a,color:#fff,stroke:#378ADD
    style Realtime fill:#7a2a1a,color:#fff,stroke:#e74c3c
```



### Comparison Table

```
┌──────────────────┬──────────────────────────┬──────────────────────────┐
│ Dimension        │ BATCH                    │ REAL-TIME                │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Speed priority   │ Accuracy over speed      │ Speed above all          │
│ Latency          │ Not critical             │ Critical (milliseconds)  │
│ Data volume      │ Large datasets at once   │ One input at a time      │
│ Cost             │ Lower — on-demand        │ Higher — always-on       │
│ Use cases        │ Overnight recs, reports, │ Chatbots, fraud detect., │
│                  │ daily sentiment analysis │ self-driving cars        │
│ AWS service      │ SageMaker Batch          │ SageMaker Real-Time      │
│                  │ Transform                │ Endpoints                │
└──────────────────┴──────────────────────────┴──────────────────────────┘
```

> **⚠️ Exam Key:** Real-time endpoints run 24/7 → higher cost. Batch spins up on demand → lower cost. Choose batch whenever latency is not critical.

---

## 6. Deep Learning and Neural Networks

### Neural Network Architecture

```
                INPUT           HIDDEN          HIDDEN          OUTPUT
                LAYER           LAYER 1         LAYER 2         LAYER
                ┌─────┐
   Age ────────►│     │
                │  N  │──┐     ┌─────┐         ┌─────┐
                └─────┘  ├────►│     │──┐       │     │──┐
                         │     │  H  │  ├──────►│  H  │  ├────► Buy: 82%
                ┌─────┐  │     └─────┘  │       └─────┘  │
  Spend ────────►│     │──┤             │                 │
                │  N  │  │     ┌─────┐  │       ┌─────┐  ├────► No:  18%
                └─────┘  ├────►│     │──┤       │     │──┘
                         │     │  H  │  ├──────►│  H  │
                ┌─────┐  │     └─────┘  │       └─────┘
 Visits ────────►│     │──┘             │
                │  N  │         ┌─────┐ │
                └─────┘    ────►│     │─┘
                                │  H  │
                                └─────┘

        Thicker connections = higher learned weights = more influence
```

```mermaid
graph LR
    subgraph Input["Input Layer\n(Raw features)"]
        I1["Age: 35"]
        I2["Spend: $200"]
        I3["Visits: 12"]
    end
    subgraph Hidden1["Hidden Layer 1\n(Simple patterns)"]
        H1["Node H1"]
        H2["Node H2"]
        H3["Node H3"]
    end
    subgraph Hidden2["Hidden Layer 2\n(Complex patterns)"]
        H4["Node H4"]
        H5["Node H5"]
    end
    subgraph Output["Output Layer\n(Prediction)"]
        O1["Buy: 82%"]
        O2["No: 18%"]
    end

    I1 --> H1 & H2 & H3
    I2 --> H1 & H2 & H3
    I3 --> H1 & H2 & H3
    H1 & H2 & H3 --> H4 & H5
    H4 & H5 --> O1 & O2

    style Input fill:#1a4a7a,color:#fff
    style Hidden1 fill:#4a3a9a,color:#fff
    style Hidden2 fill:#4a3a9a,color:#fff
    style Output fill:#7a2a1a,color:#fff
```



### How Neural Networks Learn (Backpropagation)

```mermaid
sequenceDiagram
    participant D as Training Data
    participant N as Neural Network
    participant L as Loss Function
    participant B as Backpropagation

    D->>N: Forward pass (input → prediction)
    N->>L: Compare prediction vs correct answer
    L->>B: Calculate error (loss)
    B->>N: Adjust weights backwards through layers
    Note over N,B: Repeat thousands of times until\nerror is minimized
```



### Key Neural Network Concepts


| Term                | Definition                                                                             |
| ------------------- | -------------------------------------------------------------------------------------- |
| **Node / Neuron**   | Basic processing unit — receives inputs, applies a function, passes output forward     |
| **Layer**           | Group of nodes — Input, Hidden, or Output                                              |
| **Hidden Layer**    | Intermediate layers where the network learns internal representations                  |
| **Weight**          | Strength of connection between nodes — adjusted during training                        |
| **"Deep"**          | Many hidden layers — enables learning of complex, abstract patterns                    |
| **Backpropagation** | Training mechanism — calculates error and adjusts weights backward through the network |


---

## 7. Computer Vision and NLP

### AWS Services by Domain

```mermaid
graph TD
    subgraph CV["👁️ Computer Vision"]
        R["Amazon Rekognition\n• Image classification\n• Object detection\n• Face analysis\n• Content moderation\n• Video analysis"]
        T["Amazon Textract\n• Extract text from docs\n• Form field extraction\n• Table data extraction\n• Goes beyond OCR"]
    end

    subgraph NLP["💬 Natural Language Processing"]
        C["Amazon Comprehend\n• Sentiment analysis\n• Entity recognition\n• Key phrase extraction\n• Topic modeling\n• Language detection"]
        TR["Amazon Translate\n• Neural machine translation\n• 75+ languages\n• Localize websites/apps"]
        TC["Amazon Transcribe\n• Speech → text (ASR)\n• Real-time transcription\n• Call center analytics\n• Subtitle generation"]
        P["Amazon Polly\n• Text → speech (TTS)\n• Lifelike voices\n• Dozens of languages\n• Speech-enabled apps"]
        L["Amazon Lex\n• Conversational chatbots\n• ASR + NLU\n• Intent recognition\n• Powers Amazon Alexa"]
    end

    style CV fill:#1a4a7a,color:#fff,stroke:#378ADD
    style NLP fill:#4a3a9a,color:#fff,stroke:#9b59b6
```



### NLP Services — Quick Reference

```
┌──────────────────┬────────────────────────────────────────────────────┐
│ SERVICE          │ WHAT IT DOES                                       │
├──────────────────┼────────────────────────────────────────────────────┤
│ Comprehend       │ UNDERSTANDS text — sentiment, entities, topics     │
│ Translate        │ CONVERTS between languages                         │
│ Transcribe       │ SPEECH → TEXT (ears) — ASR service                 │
│ Polly            │ TEXT → SPEECH (voice) — TTS service                │
│ Lex              │ BUILDS chatbots — ASR + intent recognition         │
│ Textract         │ EXTRACTS data from scanned documents               │
│ Kendra           │ SEARCHES across enterprise documents               │
└──────────────────┴────────────────────────────────────────────────────┘

Memory aid:
  Comprehend  = comprehend (understand)
  Transcribe  = transcribe (write down what's spoken)
  Polly       = Polly talks  (text → voice)
  Lex         = lexicon (language + intent for chatbots)
```

---

## 8. Foundation Models

### What is a Foundation Model?

> A **Foundation Model (FM)** is a large model pre-trained on internet-scale data that can be adapted to perform many different tasks — text generation, summarization, image creation, Q&A, and more.

### FM Lifecycle

```mermaid
flowchart TD
    S1["1️⃣ DATA SELECTION\nUnlabeled, internet-scale data\nText, images, code, audio\nMuch easier to obtain than labeled"]
    S2["2️⃣ PRE-TRAINING\nSelf-supervised learning\nModel auto-generates labels\nfrom data structure\nNo manual annotation needed"]
    S3["3️⃣ OPTIMIZATION\nPrompt Engineering\nRAG\nFine-tuning\nVaries in cost and complexity"]
    S4["4️⃣ EVALUATION\nBenchmarks and metrics\nDoes it meet business needs?"]
    S5["5️⃣ DEPLOYMENT\nIntegrate into apps via API\nSageMaker endpoints\nAmazon Bedrock"]
    S6["6️⃣ FEEDBACK LOOP\nMonitor performance\nCollect user feedback\nDetect bias or drift\nContinuously improve"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
    S6 -.->|"Retrain /\nFine-tune"| S3

    style S1 fill:#2c3e50,color:#fff
    style S2 fill:#1a4a7a,color:#fff
    style S3 fill:#0F6E56,color:#fff
    style S4 fill:#4a3a9a,color:#fff
    style S5 fill:#7a4a1a,color:#fff
    style S6 fill:#7a2a1a,color:#fff
```



### Self-Supervised Learning Explained

```
Traditional Supervised:   [Image] + [Label: "cat"] ──→ Model learns
Traditional Unsupervised: [Image] (no label)        ──→ Model finds patterns
Self-Supervised:          "The ___ sat on the mat"  ──→ Model predicts "cat"
                           (label generated FROM data itself)
```

> **⚠️ Exam Key:** FMs use **self-supervised learning** — this is why they can train on billions of documents without expensive manual labeling.

### Amazon Bedrock FM Providers

```
┌─────────────────┬──────────────────┬────────────────────────────┐
│ PROVIDER        │ MODEL(S)         │ STRENGTHS                  │
├─────────────────┼──────────────────┼────────────────────────────┤
│ Anthropic       │ Claude           │ Long context, reasoning    │
│ Meta            │ Llama            │ Open-source, versatile     │
│ Cohere          │ Command, Embed   │ Enterprise, embeddings     │
│ Mistral AI      │ Mistral, Mixtral │ Efficient, multilingual    │
│ Stability AI    │ Stable Diffusion │ Image generation           │
│ AI21 Labs       │ Jurassic         │ Text generation            │
│ Amazon          │ Titan            │ Text, image, embeddings    │
└─────────────────┴──────────────────┴────────────────────────────┘
```

---

## 9. Large Language Models

### How LLMs Process Text

```mermaid
flowchart LR
    A["📝 Raw Text\n'A puppy is a dog'"] --> B["✂️ Tokenization\n'A' 'puppy' 'is' 'a' 'dog'"]
    B --> C["🔢 Embeddings\nEach token → vector\nof numbers\n[0.23, -0.81, 0.45, ...]"]
    C --> D["⚙️ Transformer\nProcesses relationships\nbetween all tokens\nsimultaneously"]
    D --> E["📤 Output\nGenerated text,\nanswer, code,\nsummary..."]

    style A fill:#2c3e50,color:#fff
    style B fill:#1a4a7a,color:#fff
    style C fill:#0F6E56,color:#fff
    style D fill:#4a3a9a,color:#fff
    style E fill:#7a2a1a,color:#fff
```



### Tokens, Embeddings, Vectors

```
TOKENS:
  "A puppy is to dog as a kitten is to cat."
   ─┬─  ──┬──  ─┬  ─┬─  ─┬  ─┬  ──┬──   ─┬  ─┬─
    │     │     │   │    │   │     │        │   │
  [A] [puppy] [is] [to] [dog] [as] [kitten] [is] [cat]

EMBEDDINGS (semantic space):
  "cat"    vector: [0.82, -0.31, 0.56, ...]
  "kitten" vector: [0.79, -0.28, 0.53, ...]  ← very close to "cat"
  "dog"    vector: [0.75, -0.22, 0.48, ...]  ← close to "cat" family
  "car"    vector: [-0.42, 0.91, -0.23, ...] ← far away

  Words with similar meaning cluster together in vector space.
  This is why LLMs understand: "cat is to kitten as dog is to puppy"
```

---

## 10. Generative Model Types

```mermaid
graph TD
    subgraph LLM["📝 Large Language Models (LLMs)\nTransformer architecture"]
        LLM1["Input: Text tokens\nOutput: Text, code, summaries\nKey: Attention mechanism\nAWS: Claude, Titan on Bedrock"]
    end

    subgraph DIF["🎨 Diffusion Models\nNoise → image process"]
        DIF1["Input: Text prompt / noise\nOutput: High-quality images\nKey: Forward + reverse diffusion\nAWS: Stability AI on Bedrock"]
    end

    subgraph GAN["⚔️ GANs\nTwo-network competition"]
        GAN1["Generator: creates fake data\nDiscriminator: spots fakes\nCompete until indistinguishable\nUsed for: synthetic data, deepfakes"]
    end

    subgraph VAE["🔄 VAEs\nEncoder-decoder architecture"]
        VAE1["Encoder: compresses to latent space\nDecoder: generates new samples\nKey: Gaussian latent distribution\nUsed for: image editing, anomaly detection"]
    end

    subgraph MM["🌐 Multimodal Models\nMultiple data types"]
        MM1["Input: Text + image + audio\nOutput: Multiple modalities\nKey: Cross-modal understanding\nAWS: Claude 3 on Bedrock"]
    end

    style LLM fill:#1a4a7a,color:#fff
    style DIF fill:#4a3a9a,color:#fff
    style GAN fill:#7a2a1a,color:#fff
    style VAE fill:#0F6E56,color:#fff
    style MM fill:#7a4a1a,color:#fff
```



### Diffusion Model: Forward + Reverse Process

```
FORWARD DIFFUSION (training — add noise):
  [Clear Image] ──noise──→ [Slightly noisy] ──noise──→ ... ──→ [Pure noise]
        🖼️                        🌫️                              ❄️

REVERSE DIFFUSION (generation — remove noise):
  [Pure noise] ──denoise──→ [Less noisy] ──denoise──→ ... ──→ [Generated Image]
        ❄️                       🌫️                                🖼️
```

### GAN Architecture

```
  Random Noise ──→ [GENERATOR] ──→ Fake image ──┐
                                                  ▼
                                          [DISCRIMINATOR] ──→ Real or Fake?
                                                  ▲
                  Real training data ─────────────┘

  Training loop:
  1. Generator tries to produce images that fool the discriminator
  2. Discriminator learns to spot fakes
  3. Both improve until generator produces indistinguishable images
```

---

## 11. FM Optimization Techniques

> Three techniques to optimize FM output — they vary in **cost**, **complexity**, and whether they **change model weights**.

```mermaid
graph TD
    OPT["FM Optimization Options"]

    OPT --> PE["⚡ Prompt Engineering\nFastest & cheapest\nNo infrastructure needed"]
    OPT --> RAG["🔍 RAG\nRetrieval-Augmented Generation\nMedium cost & complexity"]
    OPT --> FT["🎓 Fine-Tuning\nMost customization\nHighest cost"]

    PE --> PE1["Changes weights? NO\nHow: craft input text\nBest for: quick guidance,\nformat control, role-play\nAWS: Amazon Bedrock"]
    RAG --> RAG1["Changes weights? NO\nHow: fetch docs at runtime,\ninject as context\nBest for: private/live data\nAWS: Bedrock Knowledge Bases"]
    FT --> FT1["Changes weights? YES\nHow: retrain on small\nlabeled dataset\nBest for: domain specialization\nAWS: Bedrock Fine-tuning"]

    style OPT fill:#2c3e50,color:#fff
    style PE fill:#0F6E56,color:#fff
    style RAG fill:#1a4a7a,color:#fff
    style FT fill:#4a3a9a,color:#fff
```



### Side-by-Side Comparison

```
┌──────────────────────┬─────────────────┬────────────────────┬──────────────────────┐
│ Dimension            │ PROMPT ENG.     │ RAG                │ FINE-TUNING          │
├──────────────────────┼─────────────────┼────────────────────┼──────────────────────┤
│ Changes weights?     │ ❌ No           │ ❌ No              │ ✅ Yes               │
│ Cost                 │ 💚 Low          │ 🟡 Medium          │ 🔴 High              │
│ Speed to implement   │ Hours           │ Days               │ Days to weeks        │
│ Data needed          │ Just a prompt   │ Document knowledge │ Labeled task dataset │
│                      │                 │ base               │                      │
│ Best for             │ Quick guidance, │ Real-time private/ │ Domain-specific      │
│                      │ format control  │ changing data      │ language and tone    │
│ AWS service          │ Amazon Bedrock  │ Bedrock Knowledge  │ Bedrock Fine-tuning  │
│                      │                 │ Bases              │                      │
└──────────────────────┴─────────────────┴────────────────────┴──────────────────────┘
```

### RAG Architecture

```mermaid
flowchart LR
    Q["👤 User Query\n'What is our\nrefund policy?'"] --> E["🔢 Embed Query\nConvert to vector"]
    E --> S["🔍 Search\nVector Database\n(Knowledge Base)"]
    S --> R["📄 Retrieve\nTop relevant\ndocuments"]
    R --> P["📝 Build Prompt\nOriginal query +\nretrieved context"]
    P --> FM["🤖 Foundation\nModel"]
    FM --> A["✅ Grounded\nAnswer"]

    style Q fill:#2c3e50,color:#fff
    style FM fill:#4a3a9a,color:#fff
    style A fill:#0F6E56,color:#fff
```



### Prompt Elements

```
You are an experienced journalist that excels at condensing long
articles into concise summaries.           ← INSTRUCTION (role + task)
──────────────────────────────────────────
Use a neutral, professional tone.          ← CONTEXT (guidance)
──────────────────────────────────────────
Text: [Long article text goes here]        ← INPUT DATA
──────────────────────────────────────────
Summarize in 2-3 sentences.                ← OUTPUT INDICATOR
```

---

## 12. AWS AI/ML Services Stack

```mermaid
graph TB
    subgraph GenAI["✨ GENERATIVE AI LAYER"]
        GB["Amazon Bedrock\nFMs via single API\n(Anthropic, Meta, Cohere,\nMistral AI, Stability AI, Amazon)"]
        GS["SageMaker JumpStart\n150+ open-source models\none-click deploy"]
        GQ["Amazon Q\nBusiness AI assistant\ntailored to your data"]
        GD["Amazon Q Developer\nAI code generation\nin the IDE"]
        GP["PartyRock\nNo-code Bedrock\nplayground"]
    end

    subgraph AIML["🤖 AI/ML SERVICES LAYER"]
        subgraph Text["Text and Docs"]
            AC["Amazon Comprehend\nNLP, sentiment,\nentities, topics"]
            AT["Amazon Translate\nLanguage translation\n75+ languages"]
            AX["Amazon Textract\nExtract text from\nscanned documents"]
            AK["Amazon Kendra\nEnterprise intelligent\nsearch"]
        end
        subgraph Speech["Speech"]
            AS["Amazon Transcribe\nSpeech → Text\n(ASR)"]
            AP["Amazon Polly\nText → Speech\n(TTS)"]
            AL["Amazon Lex\nConversational\nchatbots"]
        end
        subgraph Vision["Vision and Recs"]
            AR["Amazon Rekognition\nImage and video\nanalysis"]
            APR["Amazon Personalize\nPersonalized\nrecommendations"]
            ADR["AWS DeepRacer\nReinforcement\nlearning hands-on"]
        end
    end

    subgraph ML["⚙️ ML FRAMEWORKS LAYER"]
        SM["Amazon SageMaker AI\nBuild, train, and deploy custom ML models\nFully managed infrastructure"]
        SGT["SageMaker Ground Truth\nLabeled dataset creation"]
        SAP["SageMaker Autopilot\nAutoML — auto trains & tunes"]
        SC["SageMaker Clarify\nBias detection +\nmodel explainability"]
        SFS["SageMaker Feature Store\nStore and share ML features"]
    end

    GenAI --> AIML
    AIML --> ML

    style GenAI fill:#4a3a9a,color:#fff,stroke:#9b59b6
    style AIML fill:#1a4a7a,color:#fff,stroke:#378ADD
    style ML fill:#0F6E56,color:#fff,stroke:#2ecc71
    style Text fill:#1a3a6a,color:#fff
    style Speech fill:#1a3a6a,color:#fff
    style Vision fill:#1a3a6a,color:#fff
```



### Complete Service Reference


| Service                     | Layer          | What It Does                              | Key Exam Trigger                        |
| --------------------------- | -------------- | ----------------------------------------- | --------------------------------------- |
| **SageMaker AI**            | ML Frameworks  | Build, train, deploy custom ML models     | "custom model", "full ML workflow"      |
| **SageMaker Ground Truth**  | ML Frameworks  | Create labeled training datasets          | "label data", "annotate"                |
| **SageMaker Autopilot**     | ML Frameworks  | AutoML — auto-trains and tunes best model | "automatically find best model"         |
| **SageMaker Clarify**       | ML Frameworks  | Detect bias, explain predictions (SHAP)   | "explain prediction", "detect bias"     |
| **SageMaker Feature Store** | ML Frameworks  | Store/share features across teams         | "reuse features", "feature consistency" |
| **Amazon Bedrock**          | Generative AI  | Access multiple FMs via single API        | "foundation model", "LLM access"        |
| **SageMaker JumpStart**     | Generative AI  | One-click deploy of 150+ open models      | "quickly deploy open-source model"      |
| **Amazon Q**                | Generative AI  | Business AI assistant                     | "company data assistant"                |
| **Amazon Q Developer**      | Generative AI  | Code generation in the IDE                | "code suggestions", "IDE", "developer"  |
| **Amazon Comprehend**       | AI/ML Services | NLP — sentiment, entities, topics         | "analyze text", "sentiment analysis"    |
| **Amazon Translate**        | AI/ML Services | Language translation                      | "translate", "localize"                 |
| **Amazon Textract**         | AI/ML Services | Extract from scanned documents            | "scanned forms", "extract tables"       |
| **Amazon Kendra**           | AI/ML Services | Enterprise intelligent search             | "search company documents"              |
| **Amazon Transcribe**       | AI/ML Services | Speech → text (ASR)                       | "convert audio", "subtitles"            |
| **Amazon Polly**            | AI/ML Services | Text → speech (TTS)                       | "read text aloud", "voice app"          |
| **Amazon Lex**              | AI/ML Services | Build conversational chatbots             | "chatbot", "voice interface", "intent"  |
| **Amazon Rekognition**      | AI/ML Services | Image/video analysis                      | "images", "faces", "video analysis"     |
| **Amazon Personalize**      | AI/ML Services | Personalized recommendations              | "recommendations", "personalized"       |
| **AWS DeepRacer**           | AI/ML Services | Reinforcement learning race car           | "reinforcement learning hands-on"       |


---

## 13. Cost Considerations

```mermaid
graph LR
    subgraph Costs["💰 AWS AI/ML Cost Factors"]
        C1["Responsiveness\nHigher availability\n= Higher cost"]
        C2["Redundancy\nMulti-AZ/Region\n= More cost"]
        C3["Performance\nGPU instances\n= More cost"]
        C4["Token pricing\nBedrock, Amazon Q\nPay per token"]
        C5["Provisioned throughput\nPre-reserved capacity\n= Predictable but pricier"]
        C6["Custom models\nFine-tuning + training\n= Additional compute cost"]
    end

    style Costs fill:#2c3e50,color:#fff
```



### Cost Optimization Decision Tree

```
Need real-time response?
  ├── YES → Use SageMaker Real-Time Endpoint
  │           Consider: Serverless Inference for infrequent traffic
  └── NO  → Use SageMaker Batch Transform (cheaper, on-demand)

Optimizing an FM?
  ├── Tight budget → Prompt Engineering first (free)
  ├── Need live/changing data → RAG (no retraining cost)
  └── Need domain specialization → Fine-tuning (highest cost)

Token usage (Bedrock)?
  └── Fewer tokens in prompt = lower cost
      → Use prompt engineering to be concise
```

---

## 14. Exam Cheat Sheet

### Service Selector

```
Analyze text sentiment or entities?              → Amazon Comprehend
Extract text from a scanned document/form?       → Amazon Textract
Translate content to another language?           → Amazon Translate
Build a chatbot or voice assistant?              → Amazon Lex
Convert audio/speech to text?                   → Amazon Transcribe
Convert text to lifelike speech?                → Amazon Polly
Analyze images or videos for objects/faces?     → Amazon Rekognition
Search across company documents?                → Amazon Kendra
Personalized product recommendations?           → Amazon Personalize
Hands-on reinforcement learning?                → AWS DeepRacer
Build and train a custom ML model?              → Amazon SageMaker AI
Label training data with human reviewers?       → SageMaker Ground Truth
Explain model predictions / detect bias?        → SageMaker Clarify
Auto-find the best ML model for your data?      → SageMaker Autopilot
Access multiple foundation models via one API?  → Amazon Bedrock
Quick deploy of open-source models?             → SageMaker JumpStart
AI assistant using company's own data?          → Amazon Q
AI code generation in the IDE?                 → Amazon Q Developer
```

### Optimization Selector

```
Fastest, no cost?                       → Prompt Engineering
Private/live data, no retraining?       → RAG
Domain-specific language and tone?      → Fine-tuning
Data changes frequently?                → RAG (not fine-tuning)
Does RAG change model weights?          → NO
Does fine-tuning change model weights?  → YES
```

### Learning Type Selector

```
Labeled data, predict output?           → Supervised learning
No labels, find patterns?               → Unsupervised learning
Rewards/penalties, trial and error?     → Reinforcement learning
Small labeled + large unlabeled?        → Semi-supervised learning
```

---

*Last updated: 2026 | AWS AI Practitioner (AIF-C01) | Module 1*