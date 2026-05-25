# Module 1 — Exam Prep: 50 Practice Questions

> **Course:** AWS Artificial Intelligence Practitioner (AIF-C01)
> **Module:** Fundamentals of Machine Learning and Artificial Intelligence
> **Format:** Multiple choice — same style as the real AIF-C01 exam
> **Tip:** Try answering before reading the explanation. Focus on the reasoning, not just the answer.

---

## How to Use This File

- Each question includes **4 options**, the **correct answer**, and a **detailed explanation**
- Questions are grouped by topic to help you identify weak areas
- A **Quick Answer Key** is at the bottom for rapid review

---

## Topic 1: AI / ML / Deep Learning / Generative AI Hierarchy
*(Questions 1–7)*

---

### Question 1
Which of the following correctly describes the relationship between AI, ML, deep learning, and generative AI?

- A) AI is a subset of ML, which is a subset of deep learning
- B) ML, deep learning, and generative AI are all completely separate and unrelated fields
- C) Generative AI is a subset of deep learning, which is a subset of ML, which is a subset of AI
- D) Generative AI is the broadest category, containing ML, AI, and deep learning

**✅ Answer: C**

> **Explanation:** The correct nesting is AI (broadest) → ML (a type of AI) → Deep Learning (a type of ML) → Generative AI (a subset of deep learning). Each inner concept is a more specific approach within the one containing it.

---

### Question 2
A company describes its new software as a system that makes decisions by following a set of handcrafted "if-then" rules, with no learning from data. Which category best describes this system?

- A) Machine learning
- B) Deep learning
- C) Traditional / rule-based AI
- D) Generative AI

**✅ Answer: C**

> **Explanation:** Rule-based systems are a form of traditional AI that don't learn from data. ML requires learning from data. Deep learning and generative AI are further specializations of ML.

---

### Question 3
Amazon Rekognition can analyze millions of images and streaming videos to identify objects, people, and scenes. Which AI technology powers this capability?

- A) Traditional rule-based AI
- B) Supervised learning only
- C) Deep learning
- D) Reinforcement learning

**✅ Answer: C**

> **Explanation:** Amazon Rekognition is built on deep learning — specifically convolutional neural networks (CNNs). Deep learning excels at unstructured data like images and video.

---

### Question 4
Which of the following best describes how generative AI differs from traditional ML models?

- A) Generative AI requires more labeled data than traditional ML
- B) Generative AI can only work with text data
- C) Generative AI uses pre-trained models to create new content without retraining for each new task
- D) Generative AI is a rule-based system that doesn't require training data

**✅ Answer: C**

> **Explanation:** Generative AI adapts pre-trained foundation models to new tasks through prompting or fine-tuning — without full retraining. Traditional ML typically trains a separate model per task.

---

### Question 5
Which of the following is an example of a deep learning application in AWS?

- A) Amazon S3 for storing training data
- B) Amazon Rekognition for image and video analysis
- C) AWS CloudFormation for infrastructure deployment
- D) Amazon RDS for storing structured data

**✅ Answer: B**

> **Explanation:** Amazon Rekognition uses deep learning to analyze images and videos. S3, CloudFormation, and RDS are infrastructure services unrelated to deep learning.

---

### Question 6
A developer describes their new system as being "inspired by the human brain, using layers of interconnected nodes." Which technology are they describing?

- A) Decision tree
- B) Neural network
- C) Rule-based expert system
- D) Relational database

**✅ Answer: B**

> **Explanation:** Neural networks are the core technology of deep learning. They are modeled after the structure of the brain — interconnected nodes (neurons) organized into layers that learn from data.

---

### Question 7
Which of the following is a key advantage of generative AI over traditional supervised ML models?

- A) Generative AI is always cheaper to run
- B) A single foundation model can be adapted to many tasks without retraining from scratch
- C) Generative AI doesn't need any training data at all
- D) Generative AI only works with structured tabular data

**✅ Answer: B**

> **Explanation:** Foundation models in generative AI are trained once on massive datasets and can then be adapted to many tasks (summarization, translation, Q&A, image generation) through prompting or fine-tuning — without retraining from scratch for every use case.

---

## Topic 2: Training Data
*(Questions 8–15)*

---

### Question 8
A company has a dataset of 100,000 product images. None of the images have been tagged with categories or labels. What type of data is this?

- A) Labeled structured data
- B) Labeled unstructured data
- C) Unlabeled structured data
- D) Unlabeled unstructured data

**✅ Answer: D**

> **Explanation:** Images with no tags or annotations are **unlabeled** (no target variable). Images also have no predefined row/column format, making them **unstructured**. This is the most common type of real-world raw data.

---

### Question 9
Which phrase best summarizes why data quality is the most critical factor in building a successful ML model?

- A) More data always produces a better model
- B) Garbage in, garbage out
- C) The algorithm choice matters more than data quality
- D) Models can self-correct for bad data during training

**✅ Answer: B**

> **Explanation:** "Garbage in, garbage out" captures the fundamental principle: even the best algorithm will produce a poor model if trained on low-quality, biased, or incorrect data. Data preparation is the most critical stage of ML.

---

### Question 10
A financial analyst has a spreadsheet with columns for transaction date, amount, merchant name, and fraud label (yes/no). What type of data is this?

- A) Unlabeled unstructured data
- B) Labeled unstructured data
- C) Unlabeled structured data
- D) Labeled structured data

**✅ Answer: D**

> **Explanation:** A spreadsheet with rows and columns is **structured**. The fraud label (yes/no) makes this **labeled** data. This is a classic supervised learning dataset — the label is the target variable the model will learn to predict.

---

### Question 11
A data scientist is working with IoT sensor readings that record temperature every 5 minutes for the past year. What subtype of structured data is this?

- A) Tabular data
- B) Time-series data
- C) Text data
- D) Image data

**✅ Answer: B**

> **Explanation:** Time-series data consists of values measured at successive points in time (sensor readings every 5 minutes). While it's stored in a structured format, the temporal ordering is a key feature that distinguishes it from regular tabular data.

---

### Question 12
A model trained on customer purchase data consistently recommends products only to certain demographic groups, ignoring others. What is the most likely root cause?

- A) The neural network has too many hidden layers
- B) The inference endpoint is misconfigured
- C) The training data contains bias toward certain groups
- D) Amazon Bedrock was not used for training

**✅ Answer: C**

> **Explanation:** Biased training data is the most common cause of biased model outputs. If the training data over-represents or under-represents certain groups, the model learns and perpetuates that bias. This is why data quality and diversity are critical.

---

### Question 13
Which AWS service is specifically designed to help teams create accurately labeled training datasets using human reviewers?

- A) Amazon Textract
- B) Amazon Rekognition
- C) Amazon SageMaker Ground Truth
- D) Amazon Comprehend

**✅ Answer: C**

> **Explanation:** SageMaker Ground Truth is the AWS data labeling service. It uses human workforces (or automated labeling) to annotate images, text, audio, and video, creating labeled datasets for supervised learning.

---

### Question 14
Which of the following best describes the purpose of splitting data into training and test sets?

- A) To reduce the storage cost of the dataset in Amazon S3
- B) To evaluate the model's performance on data it has never seen during training
- C) To separate structured from unstructured data
- D) To speed up the training process by using less data

**✅ Answer: B**

> **Explanation:** The train/test split simulates real-world conditions. The model trains on one portion and is evaluated on the held-out test set to measure how well it generalizes to new, unseen data — a fundamental ML best practice.

---

### Question 15
A healthcare company needs to train a diagnostic model but only has 200 labeled X-ray images along with 20,000 unlabeled ones. Which approach allows them to use all available data?

- A) Supervised learning using only the 200 labeled images
- B) Unsupervised learning using only the 20,000 unlabeled images
- C) Semi-supervised learning combining both labeled and unlabeled data
- D) Reinforcement learning using doctor feedback as rewards

**✅ Answer: C**

> **Explanation:** Semi-supervised learning leverages a small amount of labeled data plus a large amount of unlabeled data. This is ideal when labeling is expensive (like medical imaging) but raw data is plentiful.

---

## Topic 3: ML Learning Types
*(Questions 16–22)*

---

### Question 16
An email provider trains a model to classify incoming emails as "spam" or "not spam" using thousands of previously labeled examples. Which ML type is being used?

- A) Unsupervised learning
- B) Reinforcement learning
- C) Supervised learning
- D) Self-supervised learning

**✅ Answer: C**

> **Explanation:** Supervised learning uses labeled training data — in this case, emails already labeled spam or not spam. The model learns a mapping from email features to the correct label.

---

### Question 17
A retail company has millions of customer transaction records but no predefined categories. They want the system to automatically discover natural customer groupings. Which ML type is most appropriate?

- A) Supervised learning
- B) Reinforcement learning
- C) Unsupervised learning
- D) Semi-supervised learning

**✅ Answer: C**

> **Explanation:** Unsupervised learning discovers hidden patterns and structures in unlabeled data. Customer segmentation via clustering (e.g., k-means) is a classic unsupervised learning application — no predefined categories needed.

---

### Question 18
An AI system for a video game character learns to navigate a maze by receiving +10 points when it reaches the exit and -1 for each wall collision. Which ML type does this describe?

- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Deep learning

**✅ Answer: C**

> **Explanation:** Reinforcement learning trains an agent by providing rewards and penalties for its actions. The agent learns to maximize cumulative reward over time through trial and error — no labeled examples of "correct moves" are needed.

---

### Question 19
AWS DeepRacer is an example of which type of machine learning?

- A) Supervised learning
- B) Unsupervised learning
- C) Transfer learning
- D) Reinforcement learning

**✅ Answer: D**

> **Explanation:** AWS DeepRacer uses reinforcement learning — the autonomous car learns to drive by receiving reward signals based on how well it stays on track. It's AWS's hands-on educational tool for learning reinforcement learning.

---

### Question 20
Which of the following scenarios best describes unsupervised learning?

- A) A model predicts house prices using 1,000 sold homes with known prices
- B) A model groups news articles by topic without any predefined categories
- C) A robot learns to stack boxes by getting points when it succeeds
- D) A spam filter trained on labeled spam/non-spam emails

**✅ Answer: B**

> **Explanation:** Grouping items without predefined categories is unsupervised learning (clustering). Option A is supervised regression. Option C is reinforcement learning. Option D is supervised classification.

---

### Question 21
A company has a large dataset of customer reviews. A small portion (5%) have been manually labeled as positive, negative, or neutral. They want to label the remaining 95%. Which approach leverages all available data most efficiently?

- A) Only train on the 5% labeled data
- B) Discard the labeled data and use unsupervised learning
- C) Use semi-supervised learning combining both labeled and unlabeled reviews
- D) Use reinforcement learning with customer ratings as rewards

**✅ Answer: C**

> **Explanation:** Semi-supervised learning uses a small labeled dataset alongside a large unlabeled one. It's more accurate than unsupervised alone and more scalable than labeling everything manually. SageMaker Ground Truth supports semi-supervised workflows.

---

### Question 22
Which statement correctly distinguishes supervised from unsupervised learning?

- A) Supervised learning doesn't require data; unsupervised learning does
- B) Supervised learning uses labeled data to predict outputs; unsupervised learning finds hidden patterns in unlabeled data
- C) Unsupervised learning requires more compute than supervised learning
- D) Supervised learning can only handle structured data; unsupervised can only handle unstructured data

**✅ Answer: B**

> **Explanation:** The core distinction: supervised learning requires labeled data and learns a mapping from inputs to known outputs. Unsupervised learning has no labels and discovers inherent patterns or structures in the data.

---

## Topic 4: Inferencing
*(Questions 23–27)*

---

### Question 23
A streaming music platform wants to generate personalized daily playlists for its 50 million users every morning using the previous day's listening data. Which inferencing type is most appropriate?

- A) Real-time inferencing
- B) Batch inferencing
- C) Online inferencing
- D) Streaming inferencing

**✅ Answer: B**

> **Explanation:** Generating playlists overnight for all users at once is a classic batch workload — large volume, non-time-sensitive, scheduled. Batch inferencing is more cost-effective than running always-on endpoints.

---

### Question 24
A cybersecurity company needs to analyze network packets and flag potential intrusions within 50 milliseconds. Which inferencing type is required?

- A) Batch inferencing
- B) Scheduled inferencing
- C) Real-time inferencing
- D) Asynchronous inferencing

**✅ Answer: C**

> **Explanation:** Detecting intrusions within milliseconds requires real-time inferencing — the model must respond to each individual input almost instantaneously. Latency is critical in this use case.

---

### Question 25
Which AWS SageMaker feature is designed to run inference jobs on large datasets stored in Amazon S3?

- A) SageMaker Real-Time Endpoints
- B) SageMaker Batch Transform
- C) SageMaker Autopilot
- D) SageMaker Ground Truth

**✅ Answer: B**

> **Explanation:** SageMaker Batch Transform runs batch inference on S3 datasets. It provisions resources, processes the job, saves results back to S3, and shuts down — no always-on endpoint needed, which keeps costs low.

---

### Question 26
A company is evaluating cost optimization for their ML workload. They run a model that generates monthly financial reports and a chatbot that responds to customers instantly. What is the most cost-effective configuration?

- A) Both on real-time endpoints to ensure consistency
- B) Both on batch transform to save costs
- C) Monthly reports on batch transform; chatbot on a real-time endpoint
- D) Monthly reports on a real-time endpoint; chatbot on batch transform

**✅ Answer: C**

> **Explanation:** Match the inferencing type to the requirement. Reports are time-insensitive → batch (cheaper). The chatbot needs immediate response → real-time endpoint. Mixing them as in option C is the optimal cost configuration.

---

### Question 27
Why does real-time inferencing typically cost more than batch inferencing on AWS?

- A) Real-time models are more complex and require more parameters
- B) Real-time inferencing requires always-on compute endpoints that run 24/7
- C) AWS charges a premium for faster inference speeds
- D) Batch inferencing uses spot instances automatically, reducing cost

**✅ Answer: B**

> **Explanation:** Real-time endpoints are always running, even when not actively handling requests, resulting in continuous compute charges. Batch jobs spin up compute only when processing the job — a significant cost difference for infrequent workloads.

---

## Topic 5: Deep Learning and Neural Networks
*(Questions 28–32)*

---

### Question 28
In a neural network, what is the primary function of hidden layers?

- A) To store the raw training data used during model fitting
- B) To receive the original input features from the dataset
- C) To learn increasingly complex internal representations and patterns from the data
- D) To produce the final prediction or classification output

**✅ Answer: C**

> **Explanation:** Hidden layers learn internal representations. Early hidden layers learn simple patterns (edges in images, common word pairs in text). Deeper layers combine those into complex abstract concepts. This hierarchical learning is the power of deep learning.

---

### Question 29
A neural network has an input layer, seven hidden layers, and an output layer. What does the term "deep" in deep learning refer to in this context?

- A) The complexity of the mathematical formulas inside each node
- B) The large amount of data required for training
- C) The multiple hidden layers in the network architecture
- D) The depth of the AWS infrastructure supporting the model

**✅ Answer: C**

> **Explanation:** "Deep" specifically refers to the many hidden layers in the network. More layers allow the model to learn more abstract, hierarchical feature representations. This depth is what distinguishes deep learning from shallow (single-layer) neural networks.

---

### Question 30
During neural network training, what is backpropagation responsible for?

- A) Splitting the dataset into training and test sets
- B) Calculating the prediction error and adjusting connection weights backward through the network
- C) Converting raw data into numerical vectors before feeding it into the model
- D) Deploying the trained model to a SageMaker endpoint

**✅ Answer: B**

> **Explanation:** Backpropagation is the training algorithm. After each forward pass produces a prediction, the error (loss) is calculated. Backpropagation then flows that error backward through all layers, adjusting each connection's weight to reduce future errors.

---

### Question 31
A company wants to detect whether products on an assembly line are defective by analyzing camera images in real time. Which AWS service leverages deep learning for this computer vision task?

- A) Amazon Comprehend
- B) Amazon Transcribe
- C) Amazon Rekognition
- D) Amazon Kendra

**✅ Answer: C**

> **Explanation:** Amazon Rekognition uses deep learning-based computer vision for image and video analysis. It can identify objects, detect anomalies, and analyze visual content without requiring ML expertise.

---

### Question 32
What makes deep learning particularly effective for unstructured data like images, audio, and text compared to traditional ML algorithms?

- A) Deep learning requires less training data than traditional ML
- B) Deep learning automatically discovers complex features and patterns through its layered architecture
- C) Deep learning uses rule-based logic to classify unstructured data
- D) Deep learning can only run on AWS GPU instances

**✅ Answer: B**

> **Explanation:** Traditional ML algorithms often require manual feature engineering (humans deciding what features matter). Deep learning automatically discovers relevant features through its layered neural network — early layers detect simple patterns, deeper layers combine them into complex representations. This makes it powerful for raw unstructured data.

---

## Topic 6: Computer Vision and NLP Services
*(Questions 33–39)*

---

### Question 33
A company wants to automatically analyze thousands of customer support chat logs to determine whether customers are satisfied or frustrated. Which AWS service is most appropriate?

- A) Amazon Transcribe
- B) Amazon Kendra
- C) Amazon Polly
- D) Amazon Comprehend

**✅ Answer: D**

> **Explanation:** Amazon Comprehend is the NLP service for text analysis — it performs sentiment analysis (positive, negative, neutral, mixed), entity recognition, key phrase extraction, and topic modeling. It's the go-to service for understanding text content.

---

### Question 34
A law firm needs to digitize and extract data from thousands of paper contracts, including text in tables and form fields. Which AWS service should they use?

- A) Amazon Rekognition
- B) Amazon Textract
- C) Amazon Comprehend
- D) Amazon Transcribe

**✅ Answer: B**

> **Explanation:** Amazon Textract goes beyond OCR — it extracts text, form fields, and table data from scanned documents. Unlike simple OCR that just reads text linearly, Textract understands document structure (forms, tables), making it ideal for contracts and legal documents.

---

### Question 35
A global e-commerce company wants to automatically translate their product descriptions from English into 20 different languages for regional marketplaces. Which AWS service should they use?

- A) Amazon Comprehend
- B) Amazon Polly
- C) Amazon Translate
- D) Amazon Lex

**✅ Answer: C**

> **Explanation:** Amazon Translate is the neural machine translation service. It provides high-quality, natural-sounding translation across 75+ languages and is designed for translating large volumes of content like websites and applications.

---

### Question 36
A company wants to add an intelligent virtual assistant to their customer service platform that can understand what users are saying and respond with the right action. Which AWS service handles the intent recognition component?

- A) Amazon Polly
- B) Amazon Transcribe
- C) Amazon Lex
- D) Amazon Comprehend

**✅ Answer: C**

> **Explanation:** Amazon Lex provides both ASR (automatic speech recognition to convert speech to text) AND NLU (natural language understanding to recognize user intent). It's the complete service for building conversational interfaces — the same technology powering Amazon Alexa.

---

### Question 37
A media company wants to automatically generate time-stamped transcripts for their video library to enable subtitle generation and content search. Which AWS service is most appropriate?

- A) Amazon Polly
- B) Amazon Lex
- C) Amazon Translate
- D) Amazon Transcribe

**✅ Answer: D**

> **Explanation:** Amazon Transcribe is the ASR service that converts audio/video to text with timestamps for every word. It's explicitly designed for use cases like subtitle generation, call center transcription, and making audio content searchable.

---

### Question 38
A developer is building a navigation app that should read turn-by-turn directions aloud in the user's preferred language. Which AWS service should they use?

- A) Amazon Transcribe
- B) Amazon Polly
- C) Amazon Lex
- D) Amazon Translate

**✅ Answer: B**

> **Explanation:** Amazon Polly converts text to lifelike speech (TTS). It offers dozens of voices across many languages, making it ideal for apps that need to read text aloud. Transcribe does the opposite (speech to text).

---

### Question 39
An enterprise wants their employees to search across thousands of internal documents — PDFs, wikis, and intranet pages — using natural language questions. Which AWS service provides this capability?

- A) Amazon Comprehend
- B) Amazon Textract
- C) Amazon Kendra
- D) Amazon Rekognition

**✅ Answer: C**

> **Explanation:** Amazon Kendra is the intelligent enterprise search service. It uses ML to index and search across diverse content repositories, returning relevant answers to natural language queries — even when content is scattered across multiple systems.

---

## Topic 7: Foundation Models and LLMs
*(Questions 40–44)*

---

### Question 40
What is the primary reason foundation models can be pre-trained on unlabeled internet-scale data without requiring human annotation?

- A) Foundation models are simpler than traditional ML models and don't need labels
- B) They use self-supervised learning, generating their own labels from the structure of the data
- C) AWS provides pre-labeled internet datasets through Amazon S3
- D) Foundation models skip the training phase and go directly to fine-tuning

**✅ Answer: B**

> **Explanation:** FMs use self-supervised learning — the model creates its own training signal from data structure (e.g., predicting a masked word). This eliminates the need for expensive human labeling, enabling training on billions of documents.

---

### Question 41
In the context of large language models, what are "tokens"?

- A) Security credentials used to authenticate API requests to Amazon Bedrock
- B) The basic units of text a model processes — words, subwords, or characters
- C) GPU time units used to calculate the cost of model training
- D) Version identifiers for foundation models in SageMaker JumpStart

**✅ Answer: B**

> **Explanation:** Tokens are the fundamental text units LLMs process. Text is broken into tokens (words, parts of words, or characters) before being converted to embeddings. Token count also determines cost in Bedrock's pricing model.

---

### Question 42
Two words have embedding vectors that are very close together in vector space. What does this indicate?

- A) The words have the same number of letters
- B) The words appear in the same position in training documents
- C) The words are semantically related — they have similar meanings or contexts
- D) The words were trained on the same GPU instance

**✅ Answer: C**

> **Explanation:** Embeddings capture semantic meaning. Words used in similar contexts end up with similar vector representations. For example, "cat," "kitten," and "feline" cluster near each other. This is what allows LLMs to understand language relationships without explicit rules.

---

### Question 43
Which AWS service provides access to foundation models from Anthropic, Meta, Cohere, Mistral AI, Stability AI, and Amazon through a single serverless API?

- A) Amazon SageMaker AI
- B) Amazon Kendra
- C) SageMaker JumpStart
- D) Amazon Bedrock

**✅ Answer: D**

> **Explanation:** Amazon Bedrock is the fully managed service that provides unified API access to FMs from multiple top AI providers. It's serverless — no infrastructure management required — and supports prompt engineering, RAG, and fine-tuning.

---

### Question 44
A company wants to generate photorealistic product images from text descriptions for their e-commerce platform. Which type of generative model architecture is best suited for this?

- A) Large language model (LLM)
- B) Variational autoencoder (VAE)
- C) Diffusion model
- D) Recurrent neural network (RNN)

**✅ Answer: C**

> **Explanation:** Diffusion models excel at high-quality text-to-image generation. They learn by adding noise to images (forward diffusion) and then learning to reverse the process (denoising). Stability AI on Amazon Bedrock uses this architecture.

---

## Topic 8: FM Optimization Techniques
*(Questions 45–50)*

---

### Question 45
A startup wants to improve an FM's output to match their desired format and tone. They have no budget for infrastructure and need results today. Which optimization technique should they use first?

- A) Fine-tuning with labeled examples
- B) Retrieval-augmented generation (RAG)
- C) Prompt engineering
- D) Continuous pre-training

**✅ Answer: C**

> **Explanation:** Prompt engineering is the fastest and cheapest technique — it requires no infrastructure, no training, and no model changes. Craft a better prompt to guide the model's output. The course states it is explicitly "the fastest and lowest cost option."

---

### Question 46
A legal firm wants their AI assistant to always have access to their current client agreements and case files, which are updated daily. They cannot afford to retrain the model each day. Which optimization method is most appropriate?

- A) Fine-tuning the model daily with new case files
- B) Retrieval-augmented generation (RAG), retrieving relevant documents at runtime
- C) Prompt engineering with brief summaries of each case
- D) Deploying a new foundation model weekly

**✅ Answer: B**

> **Explanation:** RAG retrieves documents from a knowledge base at inference time — no retraining needed. This makes it ideal for frequently updated content like daily case files. The model always accesses the latest documents without any retraining cost.

---

### Question 47
A medical company wants their foundation model to write clinical notes using the precise terminology and documentation style of their hospital system. The model currently uses generic language. Which optimization technique should they use?

- A) Prompt engineering, because it's the cheapest option
- B) RAG, because it can retrieve clinical terminology
- C) Fine-tuning on a dataset of clinical notes from their hospital system
- D) Increase the max token limit

**✅ Answer: C**

> **Explanation:** Fine-tuning is ideal when you need the model to deeply internalize a specific style, terminology, or domain-specific language. Training on the hospital's own clinical notes adjusts the model's weights to produce appropriate documentation style.

---

### Question 48
Which of the following statements correctly describes the difference between RAG and fine-tuning?

- A) RAG modifies the model's weights; fine-tuning retrieves external documents
- B) Fine-tuning modifies the model's weights; RAG retrieves context at runtime without changing weights
- C) Both RAG and fine-tuning modify the model's weights
- D) Neither RAG nor fine-tuning change the model's weights

**✅ Answer: B**

> **Explanation:** This is one of the most important distinctions for the exam. Fine-tuning changes model weights by further training on a task-specific dataset. RAG keeps the model weights unchanged and instead retrieves relevant documents to inject as context in the prompt at runtime.

---

### Question 49
An organization is choosing between RAG and fine-tuning to improve their customer service FM. Their knowledge base changes every week with new product information. Which approach is most practical and why?

- A) Fine-tuning — it produces better results for any use case
- B) RAG — it can access the latest documents at inference time without retraining
- C) Fine-tuning — it's cheaper than building a vector database
- D) Neither — they should use prompt engineering exclusively

**✅ Answer: B**

> **Explanation:** When data changes frequently (weekly in this case), RAG is the practical choice. Fine-tuning requires retraining every time the knowledge base changes — expensive and time-consuming. RAG just updates the document store, and the model automatically accesses the latest information.

---

### Question 50
A solutions architect is reviewing a prompt submitted to Amazon Bedrock. The prompt reads: *"You are a professional tax advisor. Using only IRS guidelines, answer the following question concisely in bullet points: [user question]"*. Which prompt elements are present?

- A) Instructions and output indicator only
- B) Context and input data only
- C) Instructions, context, and output indicator
- D) All four elements: instructions, context, input data, and output indicator

**✅ Answer: C**

> **Explanation:** Breaking down the prompt: "Answer the following question concisely" = **Instruction** (task). "You are a professional tax advisor. Using only IRS guidelines" = **Context** (guidance). "In bullet points" = **Output indicator** (format). The user question (`[user question]`) is the input data placeholder — but it's not yet filled in. So the present elements are instructions, context, and output indicator.

---

## Quick Answer Key

| Q | A | Q | A | Q | A | Q | A | Q | A |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C | 11 | B | 21 | C | 31 | C | 41 | B |
| 2 | C | 12 | C | 22 | B | 32 | B | 42 | C |
| 3 | C | 13 | C | 23 | B | 33 | D | 43 | D |
| 4 | C | 14 | B | 24 | C | 34 | B | 44 | C |
| 5 | B | 15 | C | 25 | B | 35 | C | 45 | C |
| 6 | B | 16 | C | 26 | C | 36 | C | 46 | B |
| 7 | B | 17 | C | 27 | B | 37 | D | 47 | C |
| 8 | D | 18 | C | 28 | C | 38 | B | 48 | B |
| 9 | B | 19 | D | 29 | C | 39 | C | 49 | B |
| 10 | D | 20 | B | 30 | B | 40 | B | 50 | C |

---

## Score Interpretation

| Score | Percentage | Readiness |
|---|---|---|
| 45–50 | 90–100% | 🟢 Excellent — well prepared for Module 1 |
| 38–44 | 75–89% | 🟡 Good — review explanations for missed questions |
| 30–37 | 60–74% | 🟠 Fair — revisit weak topics before moving on |
| Below 30 | < 60% | 🔴 Review the module content and retake |

---

## Topics to Review if You Missed Questions

| Missed questions | Topic to review |
|---|---|
| 1–7 | AI/ML/DL hierarchy and definitions |
| 8–15 | Training data types and preparation |
| 16–22 | ML learning types (supervised, unsupervised, reinforcement) |
| 23–27 | Batch vs. real-time inferencing |
| 28–32 | Neural networks and deep learning fundamentals |
| 33–39 | AWS AI/ML services (Comprehend, Lex, Transcribe, Polly, etc.) |
| 40–44 | Foundation models, LLMs, tokens, embeddings |
| 45–50 | Prompt engineering, RAG, fine-tuning |

---

*Last updated: 2026 | AWS AI Practitioner (AIF-C01) | Module 1 Exam Prep*
