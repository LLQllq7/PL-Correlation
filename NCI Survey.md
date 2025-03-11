# NCI Survey

*Li, Leqi*

**key words**：Code translation，Neural Code Intelligence，code generation，program

---

CodePTM：代码表征预训练模型

> **CodePTM**
>
> Code Pre-trained Models
>
> * CodeBERT
> * 大多基于LSTM、Transformer架构
> * 会利用代码的结构信息，如抽象语法树（AST）等
> * 预训练任务多样化：遮蔽语言建模（MLM）、替换标记检测（RTD）
>
> 应用场景：
>
> * 代码补全
> * 代码错误检测与修复
> * 代码风格迁移
> * 代码搜索与推荐
>
> 输入模态：
>
> * 单模态（Uni-modal）：仅代码
> * 双模态（Bi-modal）：NL-PL 对
> * 多模态（Multi-modal）：NL PL + AST

---

PL：Programming Language

AST：Abstract Syntax Tree，抽象语法树

NCI：Neural Code Intelligence，神经代码智能



## Typical Models

### **CuBERT**

Code Understanding BERT

* 仅对 python
* 构建了新的 benchmark（5分类+1定位修复）
* 局限性大，未广泛使用

直接照搬BERT-Large（24 层 16 个注意力头）；然后使用了一个处理过的 ETH Py150 数据集进行微调；还训练了一组 Word2Vec embeddings、Bi-LSTM 模型和 Transformer 模型用于比较。



### **CodeBERT**

来自BERT

> RoBERTa：Robustly Optimized BERT Approach（鲁棒性优化的BERT方法），是BERT模型的优化版
>
> * 更长的训练步数和更大的数据集
> * 移除NSP（Next Sentence Prediction）任务
> * 动态掩码（Dynamic Masking）
> * 更长的序列
> * 使用更大的Batch Size
>

* 6 个编程语言

* 同时使用 bimodal data（PL-NL Pairs）和unimodel data 进行预训练
* 预训练数据：CodeSearchNet 数据集（Python, Java, JavaScript, PHP, Ruby）
* 忽略了结构信息——AST

---

**预训练任务：**

1. Masked Language Modeling（MLM）
2. Replaced Token Detection（RTD）

> MLM：随机遮蔽输入文本中的部分词汇，并要求模型根据上下文预测这些被遮蔽的词汇，从而提升模型对语言的理解和生成能力。（80%被替换为[MASK]，10%被替换为随机词汇，剩余10%保持不变）
>
> RTD：部分输入token会被替换为从生成器采样的合理替代token，判别器需要对每个token进行分类，判断其是否被替换过。

下游任务……（标记）



### **GraphCodeBERT**

为 CodeBERT 添加代码的语法信息，使 CodePTM 可以显式学习代码的结构信息。基于数据流学习代码的表征，AST相关。

**预训练任务：**

1. Edge Prediction，即数据流图边预测，通过预测数据流图的边学习代码的结构信息
2. Node Alignment，即变量对齐，具体而言是学习数据流图中的某个 node 来自输入代码中的哪个 code token
3. MLM

对注意力机制做了修改。

---

学习数据流相较于学习 AST 本身有相当的信息损失，之后的 UniXcoder……

下游任务……



### **GPT-C**

GPT-2 模型的变体，作者构建了一个代码补全 Framework，称之为 IntelliCode Compose，并对多种编程语言进行建模。

四种语言：Python, C#, JavaScript 和 TypeScript



### **PLBART**

基于BART

指定了 baseline 模型，并将其分成了两种：

1. Training from Scratch，作者用下游任务的数据集从零开始训练了 LSTM + Attention 以及一个 Transformer。
2. Pre-trained Models，作者挑选了 RoBERTa、CodeBERT、GraphCodeBERT、GPT-2、CodeGPT（-adapted）。

具体的实验部分做了 Code Summarization、Code Generation、Code Translation 这三个生成式任务，效果自然是好的，在 Classification 方面做了两个任务：clone detection 和 vulnerability detection，在后者上 PLBART 不如基于 AST 的模型。



### **CodeT5**

* 利用代码的标识符信息，即 Identifier-aware Pre-training
* 统一预训练 Transformer 模型，完全使用了 encoder-decoder

文章发表时在 CodeXGLUE Benchmark 的若干任务上取得了 SOTA 效果



### **UniXcoder**

五个任务：

1. 代码理解任务：Clone Detection 和 Code Search
2. 代码生成任务：Code Summary 和 Code Generation
3. 自回归任务：Code Completion

为了能更加有效地利用 AST，构建了一个 one-to-one 的 mapping function，将 AST 转为一个序列结构（flattened token sequence），然后和Code Comments一同编码（对这个 mapping function 的有效性的证明在文章的附录中）

* 可同时兼容 Encoder-Only，Decoder-Only 和 Encoder-Decoder 三种模式的 CodePTM



### Codex

### CodeGen

### CodeT5+

### CodeLLaMA



**数据集**

ETH Py150（CuBERT）

CodeSearchNet（CodeBERT，GraphCodeBERT）

……



## *Survey on NCI*

> *A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond*

Section 1：Introction

Section 2：预备，process code，code相关任务

Section 3：预训练和微调的技术演变

Section 4：NL2Code，models & benchmark

Section 5：CI 和其他领域的关联

Section 6：CI 应用

Section 7：两面性讨论和未来发展

Section 8：总结



### Section 1

* Code Pre-trained Models (CodePTMs)
* Large Language Models for Code (CodeLLMs)



### Section 2

LSTM

code embeddings

**结构信息载体**

* **AST.** 结构信息。对源代码应用解析器(例如，Tree-sitter 4、pycparser 5和javalang 6)来获得。
* **Data Flow. ** 数据流。语义信息。可以从AST中提取，但层次、复杂度更低，成本更低
* **Control Flow.** 控制流。执行信息。可以使用静态分析器来构建.

**利用**

* Directly Encoding AST
* Utilizing AST Paths
* Transforming AST



#### Tasks

---

**Code-Code Tasks**

* Clone Detection 克隆检测
* Defect Detection 缺陷检测
* Code Repair：seq2seq，detection+repair
* Code Completion：token-level & line-level
* Code Translation：编程语言之间翻译

**Code-Text Task**

* Code Summarization：seq2seq，直接合成/检索关键词/检测相似代码片段查找注释
* Commit Message Generation：before and after edit，总结不同以生成git提交信息

**Text-Code Tasks**

* Code Retrieval 代码检索：使用特别指标来衡量文本和代码的相似度
* Code Generation：语义解析 -> 单一PL代码生成 -> 多个PLs
* Text2SQL

---

**benchmark**

Table 1,6,7,8



### Section 3

1. Architecture: Multi-layer 多层 Transformer 
2. Training Data: GitHub 大量未标记数据，小部分标记数据（适应下游任务）
3. Learning objectives: 自监督进行优化，但保留结构信息

![image-20240906172532439](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240906172532439.png)



#### Architecture

---

**Encoder-only**

* **Structure-free Models. **

*CuBERT*

*CodeBERT*

在下游任务上都远优于 word2vec 模型和 multi-layered bidirectional LSTMs

* **Structure-based Models.**

*GraphCodeBERT*

*SynCoBERT.*   引入对比学习 contrastive learning ，同步学习了代码嵌入和相应注释

*CODE-MVP.*   同时利用AST，data flow 和 control flow 进行对比学习

*DISCO.*  代码转换算法，生成正负样本，辨别功能上细微差异。

*SCodeR.*	

*DOBF.*

---

**Encoder-Decoder**

* 基于 BART：

*PLBART*

* 基于 T5：

*PyMT5*

*CodeT5*

> **Tasks**
>
> * span corruption 跨度损坏
>
> * identifier tagging  标识符标记
>
> * masked identifier prediction  掩码标识符预测（类似去混淆，是跨度损坏一种变体）
>
> * text ↔ code generation

> **Deobfuscation  代码反混淆**
>
> Deobfuscation是代码混淆（obfuscation）的逆过程。代码混淆是故意生成人类难以理解的源代码或机器码的过程，而代码反混淆则是将这些难以理解的代码转化为简单、可理解、直观的代码的过程。这个过程在软件安全、逆向工程等领域中具有重要意义。
>
> * 提高代码可读性
> * 逆向工程

> **Span Corruption  跨度损坏**
>
> 一种基于跨度损坏的去噪设置，它通过随机选择文本中的一段连续区域（即“span”），并将其替换为特定标记（如[MASK]）或哨兵标记（sentinel token），从而破坏原始语句。模型的任务是利用未被破坏的上下文信息来预测或生成这些被替换的部分。
>
> *技术特点*
>
> 1. 上下文学习：Span Corruption促使模型学习捕捉和利用上下文信息，这对于提高模型的语言理解和生成能力至关重要。
> 2. 泛化能力：由于该技术涉及对文本中不同位置和长度的span进行破坏，因此有助于模型在下游任务中表现出更好的泛化能力。
> 3. 混合降噪：在一些情况下，Span Corruption会与其他去噪技术（如因果语言建模、前缀语言建模等）相结合，形成混合降噪目标，以进一步提高模型的性能。
>
> *示例与实现*
>
> 以T5模型为例，其预训练任务之一就是Span Corruption。在训练过程中，模型会接收到被破坏了某些span的输入文本，并需要预测这些被替换的span的原始内容。为了实现这一目标，模型会利用Transformer结构中的encoder和decoder部分进行联合训练。



*CodeRL.*  Reinforcement Learning（using Unit Test Signals）

*SPT-Code.*  增强输入——集成线性化 integrating linearized AST

*NatGen.*  像 DBOF一样，“naturalize”——输入“unnatural”代码作为输入，生成语义等效，但“natural”的代码，即更符合人类编程的习惯和方式。

*CodeT5Mix.*  编码器与解码器混合，联合预训练，（decoders）权重共享

*AST-T5.*  structure-aware code segmentation

*JuPyT5.*  专用模型，为Jupyter Notebook，在cells 上训练，服务data science

* 基于UniLMs：

*CugLM* 

*UniXcoder.*  无损AST转化为序列，使用C4和CodeSearchNet数据集，前缀决定 encoder-decoder，encoder-only or decoder-only。

---

**Decoder-only**

> 自回归autoregressive，难以利用代码结构信息。主要围绕GPT-2开发。

*GPT-C.*  GPT-2变体

*GPT-CC*

*CodeGPT*

*CodeXGLUE*

*PyCodeGPT.*  共享相似的代码草图；采用专门的Python训练标记器，并在训练过程中判断数据质量；代码生成能力强

---

**内部机制与可解释性**

2 points：

1.  task-level inspection
2. internal mechanisms exploration in conjunction with code structure



### Section 4

#### CodeLLM Models

**CodeLLM**， Large Language Models for Code 

typical LLMs：PaLM，LaMDA， BLOOM 

常用datasets：ROOTS，Pile

![image-20240912114421246](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240912114421246.png)



* **Codex 系列**

*Evaluating Large Language Models Trained on Code*

GPT语言模型，在GitHub的公开代码上进行了微调，具有Python代码编写能力

> 一个production就是 Github Copilot

* repeated sampling 重复采样



* **CodeGen 系列**

开源CodeLLM

具有从350M到16.1B的模型范围，采用自回归变换器从自然语言和代码数据中学习，具有下一个令牌预测语言建模训练目标。

CodeGen2是CodeGen的升级版本，采用更严格的数据质量控制和混合对象的训练，支持更广泛的编程语言。CodeGen2.5采用多次训练和数据增强，证明了相对较小的CodeLLM也可以与更大的模型相媲美。CodeGen系列还被用作其他通用LLM的初始化器。



* **BigCode**

StarCoder是其中一个最具代表性的模型，拥有15.5B的规模和高效的生成能力，使用启发式过滤、手动检查和清理过程编译了包括86种编程语言的StarCoderData数据集。



* **CodeT5+**

开源的CodeLLMs系列，采用编码器-解码器架构，具有220M到16B的规模，采用“浅编码器和深解码器”的结构，支持编码器、解码器和编码器-解码器三种模式，可适应不同的下游应用。

InstructCodeT5+ 是一种针对自然语言指令的代码生成模型，使用合成指令跟随提示来提高性能。



* **CodeLLaMA**

一系列基础模型，包括7B、13B、34B和70B，通过训练500B代码令其从LLaMA2中派生而来。它还提供了针对Python和自我指导数据集的变体，以及对AI负责和安全的关注。



* **DeepSeek-Coder**

一系列开源的CodeLLMs，大小从1.3B到33B不等，通过筛选GitHub数据和构建存储库级代码数据来提高模型的能力。它还提供了继续预训练通用LLMs的版本。



* **Lemur**

在 LLaMA2 上预训练



此外，还有其他一些基于预训练的代码语言模型，如CodeGeeX2、Code-Qwen、AquilaCode和CodeGemma等。这些模型主要通过在通用LLM上进行额外的预训练、指令调整、高级工具使用能力和效率提升等方面进行改进。

---

#### Execution Feedback

**Learning with Execution Feedback** 

- CodeLLMs可以通过集成强化学习（RL）来进一步增强，利用编译器或解释器自动生成精确反馈。
- COMPCODER利用编译信号通过RL策略优化生成器和鉴别器。
- CodeRL利用代码单元测试信号在训练和推理阶段使用RL来优化模型。
- PPOCoder将CodeLLMs与Proximal Policy Optimization相结合，用于代码生成。
- RLCF通过引入一个评估生成代码质量的基础函数来进一步增强CodeLLM。
- Pangu-Coder2引入了一个RRTF框架，通过使用测试信号和人类偏好作为组合反馈来引导模型生成更高质量的代码。
- ExeDec通过将任务分解为执行子目标来改进组合泛化，逐步解决复杂任务。
- StepCoder通过将复杂任务分解为子任务的课程来创新，解决代码生成的探索和优化挑战。

---



#### Evaluation

**NL2Code**

* 基于匹配的评估方法，limited

* 基于执行的评估方法，更可靠

  检查生成的代码片段是否能够成功执行：能否通过一组单元测试

> 自动测评 OJ

**评估方法：pass@k**

生成k个代码样本，有通过则认为问题解决。但是方差较大

Chen等人提出了一种更稳定的指标：

![image-20240912153508752](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240912153508752.png)

生成n≥k个样本来计算正确样本c≤n的数量，从而得出一个无偏估计。

pass@k已成为NL2Code评估的标准实践。

---



#### **Benchmarks**

![image-20240912170915743](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240912170915743.png)

**HumanEval**

* 最初与Codex一起发布，包含164个手工编写的Python编码问题。使用测试用例(平均每个问题有7.7个测试)来验证这些问题

* 主要评估代码生成能力

* 多语言HumanEva引入了一个可扩展的自动化框架，能够将数据集从Python转换为12种不同语言的变体



**MBPP**

* Most Basic Programming Problems，评测入门级程序，包含974个简短的Python函数，每个函数都有一个英文描述、一个预定义的函数签名和三个手动编写的测试用例来验证
* 同样扩展到多语言，MBXP (Most Basic X Programming Problems，其中X = Java, Go, Ruby等)



**MultiPLE**

* 特别为多语言设计，根据TIOBE 16排名和GitHub使用频率将它们翻译成18种额外的编程语言来扩展范围
* 简化流程，减少人工干预

---

CodeLLMs性能

![image-20240912172528128](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240912172528128.png)

* 指令微调明显更优，甚至优于扩大模型



## *Vocabulary Sharing*

> *How Vocabulary Sharing Facilitates Multilingualism in LLaMA?*

为什么选 en→af 和 en →ro？



笛卡尔坐标系四象限图，来源于公式计算结果

![image-20240911163551874](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911163551874.png)

![image-20240911163614813](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911163614813.png)

* *Reciprocal Quadrant*
* *Altruistic Quadrant*
* *Stagnant Quadrant*
* Selfish Quadrant



### **Reciprocal 的同族语系**

Indo-European 印欧语系

有两种微调策略： Embed FT（推荐）和 Full FT（overly specialized）

![image-20240911165611240](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911165611240.png)

Embed FT 有效：sharing similar cocabulary and grammar rules



### **Altruistic 的微调策略**

三种微调策略：Full FT，Embedding FT，LoRA

![image-20240911200858089](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911200858089.png)

1. 对 FT 和 LoRA，Size增加，bilingual 的性能增加，但 Multl 性能降低，因为 overfit the specific language

2. 对 Embed，Size增加，并没有显著增加 bilingual 的性能，但是 Multl 性能显著提高，但也和 FT 的小 data volume 性能相近。

因此更建议选择 small dataset 和 Full FT，B和M性能都比较好。



### **Stagnant 的性能优化**

over-tokenization

扩展词汇表三种方法：BBPR，BPE，SP

![image-20240911203256903](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911203256903.png)

SP倾向于更好的效果。

over-tokenization 导致信息密度很低

“post-tokenization‘ ：去除公共前缀，可提升性能平均 2.6

![image-20240911203906324](C:\Users\Lucky Lee\AppData\Roaming\Typora\typora-user-images\image-20240911203906324.png)



get：

1. 训练过程：
2. 数据集：
3. 分析（分类）方式



## *DeepSeek-Coder*

DeepSeek-Coder是一系列开源代码模型，大小从1.3B到33B，经过2万亿个标记的训练。这些模型在高质量项目级代码语料库上进行了预训练，并使用了填空任务来增强代码生成和填充。经过广泛评估，DeepSeek-Coder在多个基准测试中不仅达到了开源代码模型的最新性能，还超过了现有的闭源模型。此外，DeepSeek-Coder模型采用宽松许可证，允许进行研究和无限制的商业使用。

> 略，弃案



## *DeepSeek-Coder-V2*

- DeepSeek-Coder-V2是开源的Mixture-of-Experts (MoE)代码语言模型，性能与GPT4-Turbo相当。
- 通过在DeepSeek-V2的中间检查点上继续预训练6万亿个token，显著提升了编码和数学推理能力。
- 支持的编程语言数量从86增加到338，上下文长度从16K扩展到128K。
- 在标准基准评估中，DeepSeek-Coder-V2在编码和数学基准测试中优于GPT4-Turbo、Claude 3 Opus和Gemini 1.5 Pro等闭源模型。

> 略，弃案



# Idea

> Vocabulary Sharing , Facilitates , Multilingualism , in LLaMA?

既然是模仿 Vocabulary Sharing 这篇文章：

1. 通读论文，看看具体是如何做：
   * 如何带动？
   * 如何测评？
   * 如何训练？
   * 具体实现？
2. 然后论文Survey：
   * 数据集？
   * 测评？
   * 当前最新结果？
   * CodePTM or CodeLLM？Which model（s）？

---

**思路：**

1. NL→PL在指定PL上性能测试统计
2. 由指定PL模型参数，向其family内其他PL微调，性能是否会存在同样四象限划分？
3. 进一步微调？Instruction Tuning？
4. downstream tasks？



**Question & Thinking：**

1. 暂时还没有找到比较新且好的 PL Classfication 的文章，或许可以根据 PL Sharing 来度量 Classfication？


2. 附录有非常多，图表和数据很细，相关工作要做到什么地步？

> 关于文章构成：整个Vocabulary Sharing 文章，主要围绕四象限的分析成文，但是附录中又附有大量详细说明和图表，这部分对论文成果的贡献有多少？到底是正文的分析和结构更重要，还是附录的数据和图表更重要？行文思路？



# Appendix

## PL Families

以下是一些常见的编程语言，并按照语言体系进行归类：

1. **C语言家族**

   - **C**: 经典的系统编程语言，广泛用于操作系统、嵌入式系统和底层开发。
   - **C++**: C的扩展，支持面向对象编程，广泛应用于游戏开发、系统编程。
   - **Objective-C**: C的面向对象扩展，主要用于早期的macOS和iOS开发。
   - **C#**: 微软开发的面向对象语言，基于C和C++，主要用于Windows平台和. NET框架。

2. **Java家族**

   - **Java**: 面向对象语言，跨平台性强，广泛用于企业应用和Android开发。
   - **Scala**: 运行在Java虚拟机（JVM）上，融合了面向对象和函数式编程。
   - **Kotlin**: 现代化的JVM语言，官方支持Android开发。

3. **JavaScript家族**

   - **JavaScript**: 动态脚本语言，广泛用于网页前端开发。
   - **TypeScript**: JavaScript的超集，支持静态类型检查，增强了代码的可维护性。
   - **Node.js**: JavaScript的后端运行环境，用于构建高效的网络应用。

4. **Python家族**

   - **Python**: 高级动态语言，语法简洁，广泛应用于数据科学、机器学习、Web开发、自动化等领域。
   - **Jython**: 在JVM上运行的Python实现。
   - **PyPy**: Python的高性能实现，主要用于提高Python的执行效率。

5. **Lisp家族**

   - **Common Lisp**: 多范式编程语言，广泛用于人工智能和计算机科学研究。
   - **Scheme**: Lisp的简化版，适合教学用途。
   - **Clojure**: 现代化的Lisp语言，运行在JVM上，强调不可变数据结构和并发编程。

6. **ML家族**

   - **Standard ML (SML)**: 静态类型的函数式编程语言，广泛用于编译器开发和编程语言研究。
   - **OCaml**: ML的扩展，支持面向对象编程，常用于金融和科学计算。
   - **F#**: 基于OCaml的函数式编程语言，主要用于. NET框架。

7. **Ruby家族**

   - **Ruby**: 面向对象的动态语言，强调简洁和开发者友好性，广泛用于Web开发（如Ruby on Rails）。
   - **JRuby**: 运行在JVM上的Ruby实现。

8. **Perl家族**

   - **Perl**: 动态脚本语言，适合文本处理、系统管理和Web开发。
   - **Raku**: Perl 6，Perl语言的演进版，但与Perl 5并不完全兼容。

9. **Pascal家族**

   - **Pascal**: 结构化编程语言，早期用于教学和系统编程。
   - **Delphi**: 基于Object Pascal的语言，用于Windows应用程序开发。

10. **系统编程语言**

   - **Rust**: 安全的系统编程语言，替代C和C++，广泛应用于操作系统、嵌入式系统等。
   - **Go (Golang)**: 谷歌开发的语言，适合并发编程，广泛用于网络服务、分布式系统。

11. **脚本语言**

   - **PHP**: 动态语言，广泛用于Web开发。
   - **Lua**: 轻量级脚本语言，常用于游戏开发、嵌入式系统。

12. **函数式编程语言**

   - **Haskell**: 纯函数式语言，强调不可变数据和高阶函数，常用于学术研究和金融领域。
   - **Erlang**: 面向并发和分布式系统的函数式语言，广泛用于电信行业。

13. **数据科学与统计语言**

   - **R**: 专门用于统计分析和数据可视化的语言，广泛用于数据科学。
   - **Julia**: 高性能语言，适合数值计算和数据科学领域。

14. **其他语言**

   - **Swift**: 苹果开发的语言，主要用于iOS和macOS应用开发。
   - **Dart**: 谷歌开发的语言，主要用于Flutter框架，构建跨平台移动和Web应用。
   - **V**: 现代化的系统编程语言，语法简单，强调编译速度。

这个列表涵盖了主流的编程语言及其所属的体系或家族，便于理解各个语言在技术生态中的定位。

