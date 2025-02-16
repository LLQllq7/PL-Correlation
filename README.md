# README

*Leqi Li, The University of Science & Technology of China*

> *This project is currently in progress.* 
>
> *This project is a key project under my university's 'Undergraduate Research Program'.*
>
> *Time frame: September 2024 – Present*



## Project title

*Analysis of Programming Language Correlation in Code Generation Capabilities of Large Language Models*



## Research Background

### 1 Evolution of Code Generation Language Models

In recent years, the rapid advancements in deep learning have led to significant breakthroughs in generative pre-trained language models for natural language processing (NLP). These models have also expanded into the field of code generation, with various architectures and training strategies playing a crucial role in enhancing code generation capabilities.

#### Early Exploration and CodePTM Model

Early research on code generation primarily relied on RNN-based sequence-to-sequence (Seq2Seq) models, which could generate code snippets within a limited scope. However, due to the weakness of RNNs in capturing long-range dependencies, these models struggled with complex programming tasks. The introduction of the Transformer architecture significantly improved code generation performance.

CodePTM (Pre-trained Transformer for Programming Languages) was a milestone model specifically designed for code generation tasks. Built on a pure Transformer architecture and pre-trained on multilingual code corpora, it effectively captured the semantic and structural characteristics of code. CodePTM marked a step towards unified multilingual support in code generation models.

#### CodeLLM Advancements and Technical Evolution

The CodeLLM (Code Large Language Model) series further pushed the boundaries of code generation. By increasing the scale of pre-training data and adopting deeper and wider architectures, these models significantly improved adaptability across multiple languages and tasks. For example, Codex, based on the GPT series, was fine-tuned on a large-scale code corpus, achieving remarkable performance in code completion, automatic documentation, and bug fixing.

CodeLLM models typically fall into three main categories:

- **Encoder-Only Models**: E.g., CodeBERT, primarily used for code representation tasks such as search and classification.
- **Decoder-Only Models**: E.g., Codex, focused on generative tasks such as code completion and multi-step reasoning.
- **Encoder-Decoder Models**: E.g., CodeT5, which integrates both encoding and generation capabilities for a broader range of code understanding and generation scenarios.

The flexibility of Transformer architectures enables them to adapt to various tasks while laying the foundation for large-scale, multilingual code generation models.

#### Rise of Llama Series Models

The Llama (Large Language Model Meta AI) series has set new benchmarks in both NLP and code generation with its efficient parameter utilization and outstanding performance. From Llama1 to Llama2 and the upcoming Llama3, these models continue to improve in architecture, data scaling, and training strategies, forming a strong foundation for general-purpose AI models.

In code generation tasks, Llama models benefit from their extensive multilingual pre-training, making them well-suited for cross-language transfer learning. However, fine-tuning them for specific programming languages while exploring the relationships between different languages remains an open research question.



### 2 Cross-Language Correlation in Natural Language

The NLP field has long studied cross-linguistic relationships in translation and transfer learning. The paper *How Vocabulary Sharing Facilitates Multilingualism in LLaMA?* offers valuable insights, demonstrating that shared vocabulary across related languages enhances knowledge transfer and improves translation performance.

Key findings include:

- Shared vocabulary design helps capture common linguistic features across languages.
- Fine-tuning on one language can improve zero-shot learning on related languages.
- Linguistic similarity is crucial for training and evaluating multilingual models.

This research raises an important question for code generation: Do programming languages exhibit similar cross-language relationships? For example, can Python and Java demonstrate enhanced transferability due to their syntactic and semantic similarities? Investigating this hypothesis could provide new perspectives on multilingual code generation.

------



## Research Objectives and Significance

This study aims to explore the multilingual code generation capabilities of Llama3 and the relationships between different programming languages. The key objectives include:

1. Analyzing the performance variations of Llama3 in code generation before and after fine-tuning on specific programming languages.
2. Exploring language similarity features and their impact on transfer learning in code generation.
3. Providing insights for designing multilingual code generation models.

### Research Significance

- **Theoretical Significance**:
  - Uncovering the relationships between programming languages to provide new perspectives for multilingual code generation research.
  - Offering insights into programming language classification based on generative model performance.
- **Practical Significance**:
  - Providing empirical evidence for building efficient cross-language transfer models in code generation.
  - Promoting the application of multilingual code generation models in real-world software development.

------



## Research Methodology

### 1 Model Selection and Fine-Tuning

- Use Llama3 as the base model and fine-tune it on the CodeSearchNet dataset for a specific programming language (e.g., Python).
- Evaluate code generation performance before and after fine-tuning using the HumanEval benchmark.

### 2 Cross-Language Experiments

- Apply the fine-tuned model to another programming language (e.g., Java) and assess its code generation capabilities.
- Compare performance changes before and after fine-tuning to analyze cross-language relationships.

### 3 Result Analysis and Visualization

- Use quantitative metrics (e.g., BLEU, CodeBLEU) and qualitative methods (e.g., case studies) for performance evaluation.
- Visualize cross-language relationships using heatmaps or network graphs.

------

This study will contribute to a deeper understanding of multilingual code generation and the underlying relationships between programming languages. By fine-tuning Llama3 and systematically analyzing transferability, we aim to enhance the effectiveness of code generation models across different programming languages.