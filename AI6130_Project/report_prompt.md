1. Prompt Design
To systematically evaluate the reasoning capabilities of small language models, we designed a unified prompt factory that generates specific instructional contexts. All prompts share a common base structure that injects metadata—specifically the dataset name and the expected answer type—as a header before presenting the problem. This ensures the model is grounded in the specific task domain.

Building upon this base structure, we engineered three distinct prompt variations for our ablation study:

Zero-shot (zero_shot_v1): Serves as our baseline. The system role is defined simply as a "helpful assistant," and the instruction explicitly forces the model to "Give the final answer only" without any intermediate reasoning.

Chain-of-Thought (cot_v1): Designed to elicit logical deductions. The system prompt instructs the model to be a "careful assistant," and the user instruction explicitly requests the model to "Solve step-by-step" before appending the final answer on a new line starting with "Answer:". This allows us to extract the final prediction reliably using simple string parsing.

Constrained Format (constrained_json_v1): Aims to test the models' ability to adhere to strict programmatic output schemas. The prompt restricts the output to a "single-line JSON object with exactly one key 'answer'" and provides specific formatting rules alongside a few-shot JSON example (e.g., {"answer":"42"}).

2. Prompt Comparison Table
The following table summarizes the Exact-Match (EM) accuracy of three different small language models evaluated across three datasets under our distinct prompt variations.
**Table 1: Accuracy (%) across different prompt configurations and datasets.**

| Model | Dataset | Zero-shot | CoT | Constrained (JSON) |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B-Instruct** | AQuA-RAT | 5.91% | **48.43%** | 16.93% |
| | GSM8K | 34.00% | **67.67%** | 4.00% |
| | SVAMP | 47.67% | **76.00%** | 10.00% |
| **OpenLLaMA-3B** | AQuA-RAT | 12.99% | 11.42% | **12.99%** |
| | GSM8K | 0.00% | 1.00% | **1.67%** |
| | SVAMP | 1.33% | 3.67% | **4.67%** |
| **TinyLlama-1.1B-Chat** | AQuA-RAT | **21.65%** | 21.26% | 1.97% |
| | GSM8K | **3.33%** | 1.00% | 0.33% |
| | SVAMP | **6.67%** | 4.67% | 0.33% |


3. Ablation Analysis
The prompt ablation study reveals several critical insights regarding how small language models interact with instructional designs:

The Transformative Power of CoT for Instruction-Tuned Models:
The addition of the Chain-of-Thought prompt profoundly impacted the instruction-tuned Qwen2.5-1.5B model. By simply allowing the model to generate intermediate reasoning steps, accuracy on GSM8K doubled from 34.00% to 67.67%, and SVAMP accuracy surged from 47.67% to 76.00%. This confirms that the internal logical capabilities of modern small models are robust, but they require the "computational buffer" provided by CoT generation to accurately resolve multi-step mathematical problems.

The "Distraction" of Strict Formatting Constraints:
Conversely, the constrained_json prompt caused a catastrophic degradation in performance for the Qwen model. For instance, SVAMP accuracy plummeted from 76.00% (under CoT) and 47.67% (Zero-shot) down to a mere 10.00%. Qualitative review of the outputs suggests that forcing a small model to strictly adhere to JSON syntax acts as a cognitive distractor. The model dedicates its limited capacity to formatting rules rather than mathematical logic. Furthermore, the JSON constraint inherently suppresses the step-by-step reasoning trace, mirroring the limitations seen in the zero-shot approach but with the added penalty of syntax overhead.

Model Architecture vs. Prompt Sensitivity:
The base model (OpenLLaMA-3B) and the heavily compressed chat model (TinyLlama-1.1B) exhibited distinct behaviors compared to Qwen. OpenLLaMA-3B completely failed on GSM8K in the zero-shot setting (0.00%) and saw negligible improvement with CoT (1.00%), indicating that without specific instruction-tuning, prompt engineering alone cannot inject reasoning capabilities into base models. TinyLlama-1.1B paradoxically performed best in the Zero-shot setting across all datasets, suggesting that its alignment may have been too weak to follow complex multi-step instructions, causing it to hallucinate or truncate when prompted with CoT or JSON constraints.

Ultimately, this ablation demonstrates that while CoT is highly effective, it is strictly bottlenecked by the model's pre-existing instruction-following alignment. Furthermore, developers must be cautious when imposing strict output formats (like JSON) on small models for reasoning tasks, as it actively harms predictive accuracy.