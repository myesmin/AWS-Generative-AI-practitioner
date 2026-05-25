# AWS AI Practitioner (AIF-C01)

## Module 1 Practice Exam — Fundamentals of Machine Learning and Artificial Intelligence

---

**Total Questions:** 50
**Time Suggested:** 60 minutes
**Instructions:** Choose the single best answer for each question. Do not refer to notes during the exam. Record your answers on paper or in a separate file, then check against the answer key in `module1-exam-prep-50-questions.md`.

---

## Section 1: AI / ML / Deep Learning / Generative AI Hierarchy

*Questions 1–7*

---

**Question 1**

Which of the following correctly describes the relationship between AI, ML, deep learning, and generative AI?

- A) AI is a subset of ML, which is a subset of deep learning
- B) ML, deep learning, and generative AI are all completely separate and unrelated fields
- C) Generative AI is a subset of deep learning, which is a subset of ML, which is a subset of AI
- D) Generative AI is the broadest category, containing ML, AI, and deep learning

---

**Question 2**

A company describes its new software as a system that makes decisions by following a set of handcrafted "if-then" rules, with no learning from data. Which category best describes this system?

- A) Machine learning
- B) Deep learning
- C) Traditional / rule-based AI
- D) Generative AI

---

**Question 3**

Amazon Rekognition can analyze millions of images and streaming videos to identify objects, people, and scenes. Which AI technology powers this capability?

- A) Traditional rule-based AI
- B) Supervised learning only
- C) Deep learning
- D) Reinforcement learning

---

**Question 4**

Which of the following best describes how generative AI differs from traditional ML models?

- A) Generative AI requires more labeled data than traditional ML
- B) Generative AI can only work with text data
- C) Generative AI uses pre-trained models to create new content without retraining for each new task
- D) Generative AI is a rule-based system that doesn't require training data

---

**Question 5**

Which of the following is an example of a deep learning application in AWS?

- A) Amazon S3 for storing training data
- B) Amazon Rekognition for image and video analysis
- C) AWS CloudFormation for infrastructure deployment
- D) Amazon RDS for storing structured data

---

**Question 6**

A developer describes their new system as being "inspired by the human brain, using layers of interconnected nodes." Which technology are they describing?

- A) Decision tree
- B) Neural network
- C) Rule-based expert system
- D) Relational database

---

**Question 7**

Which of the following is a key advantage of generative AI over traditional supervised ML models?

- A) Generative AI is always cheaper to run
- B) A single foundation model can be adapted to many tasks without retraining from scratch
- C) Generative AI doesn't need any training data at all
- D) Generative AI only works with structured tabular data

---

## Section 2: Training Data

*Questions 8–15*

---

**Question 8**

A company has a dataset of 100,000 product images. None of the images have been tagged with categories or labels. What type of data is this?

- A) Labeled structured data
- B) Labeled unstructured data
- C) Unlabeled structured data
- D) Unlabeled unstructured data

---

**Question 9**

Which phrase best summarizes why data quality is the most critical factor in building a successful ML model?

- A) More data always produces a better model
- B) Garbage in, garbage out
- C) The algorithm choice matters more than data quality
- D) Models can self-correct for bad data during training

---

**Question 10**

A financial analyst has a spreadsheet with columns for transaction date, amount, merchant name, and a fraud label (yes/no). What type of data is this?

- A) Unlabeled unstructured data
- B) Labeled unstructured data
- C) Unlabeled structured data
- D) Labeled structured data

---

**Question 11**

A data scientist is working with IoT sensor readings that record temperature every 5 minutes for the past year. What subtype of structured data is this?

- A) Tabular data
- B) Time-series data
- C) Text data
- D) Image data

---

**Question 12**

A model trained on customer purchase data consistently recommends products only to certain demographic groups, ignoring others. What is the most likely root cause?

- A) The neural network has too many hidden layers
- B) The inference endpoint is misconfigured
- C) The training data contains bias toward certain groups
- D) Amazon Bedrock was not used for training

---

**Question 13**

Which AWS service is specifically designed to help teams create accurately labeled training datasets using human reviewers?

- A) Amazon Textract
- B) Amazon Rekognition
- C) Amazon SageMaker Ground Truth
- D) Amazon Comprehend

---

**Question 14**

Which of the following best describes the purpose of splitting data into training and test sets?

- A) To reduce the storage cost of the dataset in Amazon S3
- B) To evaluate the model's performance on data it has never seen during training
- C) To separate structured from unstructured data
- D) To speed up the training process by using less data

---

**Question 15**

A healthcare company needs to train a diagnostic model but only has 200 labeled X-ray images along with 20,000 unlabeled ones. Which approach allows them to use all available data most effectively?

- A) Supervised learning using only the 200 labeled images
- B) Unsupervised learning using only the 20,000 unlabeled images
- C) Semi-supervised learning combining both labeled and unlabeled data
- D) Reinforcement learning using doctor feedback as rewards

---

## Section 3: ML Learning Types

*Questions 16–22*

---

**Question 16**

An email provider trains a model to classify incoming emails as "spam" or "not spam" using thousands of previously labeled examples. Which ML type is being used?

- A) Unsupervised learning
- B) Reinforcement learning
- C) Supervised learning
- D) Self-supervised learning

---

**Question 17**

A retail company has millions of customer transaction records but no predefined categories. They want the system to automatically discover natural customer groupings. Which ML type is most appropriate?

- A) Supervised learning
- B) Reinforcement learning
- C) Unsupervised learning
- D) Semi-supervised learning

---

**Question 18**

An AI system for a video game character learns to navigate a maze by receiving +10 points when it reaches the exit and -1 for each wall collision. Which ML type does this describe?

- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) Deep learning

---

**Question 19**

AWS DeepRacer is an example of which type of machine learning?

- A) Supervised learning
- B) Unsupervised learning
- C) Transfer learning
- D) Reinforcement learning

---

**Question 20**

Which of the following scenarios best describes unsupervised learning?

- A) A model predicts house prices using 1,000 sold homes with known prices
- B) A model groups news articles by topic without any predefined categories
- C) A robot learns to stack boxes by getting points when it succeeds
- D) A spam filter trained on labeled spam/non-spam emails

---

**Question 21**

A company has a large dataset of customer reviews. A small portion (5%) have been manually labeled as positive, negative, or neutral. They want to label the remaining 95%. Which approach leverages all available data most efficiently?

- A) Only train on the 5% labeled data
- B) Discard the labeled data and use unsupervised learning
- C) Use semi-supervised learning combining both labeled and unlabeled reviews
- D) Use reinforcement learning with customer ratings as rewards

---

**Question 22**

Which statement correctly distinguishes supervised from unsupervised learning?

- A) Supervised learning doesn't require data; unsupervised learning does
- B) Supervised learning uses labeled data to predict outputs; unsupervised learning finds hidden patterns in unlabeled data
- C) Unsupervised learning requires more compute than supervised learning
- D) Supervised learning can only handle structured data; unsupervised can only handle unstructured data

---

## Section 4: Inferencing

*Questions 23–27*

---

**Question 23**

A streaming music platform wants to generate personalized daily playlists for its 50 million users every morning using the previous day's listening data. Which inferencing type is most appropriate?

- A) Real-time inferencing
- B) Batch inferencing
- C) Online inferencing
- D) Streaming inferencing

---

**Question 24**

A cybersecurity company needs to analyze network packets and flag potential intrusions within 50 milliseconds. Which inferencing type is required?

- A) Batch inferencing
- B) Scheduled inferencing
- C) Real-time inferencing
- D) Asynchronous inferencing

---

**Question 25**

Which AWS SageMaker feature is designed to run inference jobs on large datasets stored in Amazon S3?

- A) SageMaker Real-Time Endpoints
- B) SageMaker Batch Transform
- C) SageMaker Autopilot
- D) SageMaker Ground Truth

---

**Question 26**

A company runs a model that generates monthly financial reports and a chatbot that responds to customers instantly. What is the most cost-effective infrastructure configuration?

- A) Both on real-time endpoints to ensure consistency
- B) Both on batch transform to save costs
- C) Monthly reports on batch transform; chatbot on a real-time endpoint
- D) Monthly reports on a real-time endpoint; chatbot on batch transform

---

**Question 27**

Why does real-time inferencing typically cost more than batch inferencing on AWS?

- A) Real-time models are more complex and require more parameters
- B) Real-time inferencing requires always-on compute endpoints that run 24/7
- C) AWS charges a premium for faster inference speeds
- D) Batch inferencing uses spot instances automatically, reducing cost

---

## Section 5: Deep Learning and Neural Networks

*Questions 28–32*

---

**Question 28**

In a neural network, what is the primary function of hidden layers?

- A) To store the raw training data used during model fitting
- B) To receive the original input features from the dataset
- C) To learn increasingly complex internal representations and patterns from the data
- D) To produce the final prediction or classification output

---

**Question 29**

A neural network has an input layer, seven hidden layers, and an output layer. What does the term "deep" in deep learning refer to in this context?

- A) The complexity of the mathematical formulas inside each node
- B) The large amount of data required for training
- C) The multiple hidden layers in the network architecture
- D) The depth of the AWS infrastructure supporting the model

---

**Question 30**

During neural network training, what is backpropagation responsible for?

- A) Splitting the dataset into training and test sets
- B) Calculating the prediction error and adjusting connection weights backward through the network
- C) Converting raw data into numerical vectors before feeding it into the model
- D) Deploying the trained model to a SageMaker endpoint

---

**Question 31**

A company wants to detect whether products on an assembly line are defective by analyzing camera images in real time. Which AWS service leverages deep learning for this computer vision task?

- A) Amazon Comprehend
- B) Amazon Transcribe
- C) Amazon Rekognition
- D) Amazon Kendra

---

**Question 32**

What makes deep learning particularly effective for unstructured data like images, audio, and text compared to traditional ML algorithms?

- A) Deep learning requires less training data than traditional ML
- B) Deep learning automatically discovers complex features and patterns through its layered architecture
- C) Deep learning uses rule-based logic to classify unstructured data
- D) Deep learning can only run on AWS GPU instances

---

## Section 6: Computer Vision and NLP Services

*Questions 33–39*

---

**Question 33**

A company wants to automatically analyze thousands of customer support chat logs to determine whether customers are satisfied or frustrated. Which AWS service is most appropriate?

- A) Amazon Transcribe
- B) Amazon Kendra
- C) Amazon Polly
- D) Amazon Comprehend

---

**Question 34**

A law firm needs to digitize and extract data from thousands of paper contracts, including text in tables and form fields. Which AWS service should they use?

- A) Amazon Rekognition
- B) Amazon Textract
- C) Amazon Comprehend
- D) Amazon Transcribe

---

**Question 35**

A global e-commerce company wants to automatically translate their product descriptions from English into 20 different languages for regional marketplaces. Which AWS service should they use?

- A) Amazon Comprehend
- B) Amazon Polly
- C) Amazon Translate
- D) Amazon Lex

---

**Question 36**

A company wants to add an intelligent virtual assistant to their customer service platform that can understand what users are saying and respond with the right action. Which AWS service handles the intent recognition component?

- A) Amazon Polly
- B) Amazon Transcribe
- C) Amazon Lex
- D) Amazon Comprehend

---

**Question 37**

A media company wants to automatically generate time-stamped transcripts for their video library to enable subtitle generation and content search. Which AWS service is most appropriate?

- A) Amazon Polly
- B) Amazon Lex
- C) Amazon Translate
- D) Amazon Transcribe

---

**Question 38**

A developer is building a navigation app that should read turn-by-turn directions aloud in the user's preferred language. Which AWS service should they use?

- A) Amazon Transcribe
- B) Amazon Polly
- C) Amazon Lex
- D) Amazon Translate

---

**Question 39**

An enterprise wants their employees to search across thousands of internal documents — PDFs, wikis, and intranet pages — using natural language questions. Which AWS service provides this capability?

- A) Amazon Comprehend
- B) Amazon Textract
- C) Amazon Kendra
- D) Amazon Rekognition

---

## Section 7: Foundation Models and LLMs

*Questions 40–44*

---

**Question 40**

What is the primary reason foundation models can be pre-trained on unlabeled internet-scale data without requiring human annotation?

- A) Foundation models are simpler than traditional ML models and don't need labels
- B) They use self-supervised learning, generating their own labels from the structure of the data
- C) AWS provides pre-labeled internet datasets through Amazon S3
- D) Foundation models skip the training phase and go directly to fine-tuning

---

**Question 41**

In the context of large language models, what are "tokens"?

- A) Security credentials used to authenticate API requests to Amazon Bedrock
- B) The basic units of text a model processes — words, subwords, or characters
- C) GPU time units used to calculate the cost of model training
- D) Version identifiers for foundation models in SageMaker JumpStart

---

**Question 42**

Two words have embedding vectors that are very close together in vector space. What does this indicate?

- A) The words have the same number of letters
- B) The words appear in the same position in training documents
- C) The words are semantically related — they have similar meanings or contexts
- D) The words were trained on the same GPU instance

---

**Question 43**

Which AWS service provides access to foundation models from Anthropic, Meta, Cohere, Mistral AI, Stability AI, and Amazon through a single serverless API?

- A) Amazon SageMaker AI
- B) Amazon Kendra
- C) SageMaker JumpStart
- D) Amazon Bedrock

---

**Question 44**

A company wants to generate photorealistic product images from text descriptions for their e-commerce platform. Which type of generative model architecture is best suited for this?

- A) Large language model (LLM)
- B) Variational autoencoder (VAE)
- C) Diffusion model
- D) Recurrent neural network (RNN)

---

## Section 8: FM Optimization Techniques

*Questions 45–50*

---

**Question 45**

A startup wants to improve an FM's output to match their desired format and tone. They have no budget for infrastructure and need results today. Which optimization technique should they use first?

- A) Fine-tuning with labeled examples
- B) Retrieval-augmented generation (RAG)
- C) Prompt engineering
- D) Continuous pre-training

---

**Question 46**

A legal firm wants their AI assistant to always have access to their current client agreements and case files, which are updated daily. They cannot afford to retrain the model each day. Which optimization method is most appropriate?

- A) Fine-tuning the model daily with new case files
- B) Retrieval-augmented generation (RAG), retrieving relevant documents at runtime
- C) Prompt engineering with brief summaries of each case
- D) Deploying a new foundation model weekly

---

**Question 47**

A medical company wants their foundation model to write clinical notes using the precise terminology and documentation style of their hospital system. The model currently uses generic language. Which optimization technique should they use?

- A) Prompt engineering, because it's the cheapest option
- B) RAG, because it can retrieve clinical terminology at runtime
- C) Fine-tuning on a dataset of clinical notes from their hospital system
- D) Increase the max token limit

---

**Question 48**

Which of the following statements correctly describes the difference between RAG and fine-tuning?

- A) RAG modifies the model's weights; fine-tuning retrieves external documents
- B) Fine-tuning modifies the model's weights; RAG retrieves context at runtime without changing weights
- C) Both RAG and fine-tuning modify the model's weights
- D) Neither RAG nor fine-tuning change the model's weights

---

**Question 49**

An organization is choosing between RAG and fine-tuning to improve their customer service FM. Their knowledge base changes every week with new product information. Which approach is most practical and why?

- A) Fine-tuning — it produces better results for any use case
- B) RAG — it can access the latest documents at inference time without retraining
- C) Fine-tuning — it's cheaper than building a vector database
- D) Neither — they should use prompt engineering exclusively

---

**Question 50**

A solutions architect reviews a prompt submitted to Amazon Bedrock: *"You are a professional tax advisor. Using only IRS guidelines, answer the following question concisely in bullet points: [user question]"*. Which prompt elements are present?

- A) Instructions and output indicator only
- B) Context and input data only
- C) Instructions, context, and output indicator
- D) All four elements: instructions, context, input data, and output indicator

---

## Answer Sheet

*Record your answers below before checking the answer key.*


| Q   | Your Answer | Q   | Your Answer | Q   | Your Answer | Q   | Your Answer | Q   | Your Answer |
| --- | ----------- | --- | ----------- | --- | ----------- | --- | ----------- | --- | ----------- |
| 1   |             | 11  |             | 21  |             | 31  |             | 41  |             |
| 2   |             | 12  |             | 22  |             | 32  |             | 42  |             |
| 3   |             | 13  |             | 23  |             | 33  |             | 43  |             |
| 4   |             | 14  |             | 24  |             | 34  |             | 44  |             |
| 5   |             | 15  |             | 25  |             | 35  |             | 45  |             |
| 6   |             | 16  |             | 26  |             | 36  |             | 46  |             |
| 7   |             | 17  |             | 27  |             | 37  |             | 47  |             |
| 8   |             | 18  |             | 28  |             | 38  |             | 48  |             |
| 9   |             | 19  |             | 29  |             | 39  |             | 49  |             |
| 10  |             | 20  |             | 30  |             | 40  |             | 50  |             |


---

## Score Tracker


| Score    | Percentage | Readiness                                 |
| -------- | ---------- | ----------------------------------------- |
| 45–50    | 90–100%    | 🟢 Excellent — well prepared for Module 1 |
| 38–44    | 75–89%     | 🟡 Good — review missed questions         |
| 30–37    | 60–74%     | 🟠 Fair — revisit weak topics             |
| Below 30 | < 60%      | 🔴 Review module content and retake       |


> **Answer Key:** See `module1-practice-questions-solutions.md`

---

*AWS AI Practitioner (AIF-C01) | Module 1 Practice Exam | 2026*