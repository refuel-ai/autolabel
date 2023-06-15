
## Benchmarking LLMs for data labeling


Key takeaways from our [technical report](https://www.refuel.ai/blog-posts/llm-labeling-technical-report):

* State of the art LLMs can label text datasets at the same or better quality compared to skilled human annotators, **but ~20x faster and ~7x cheaper**.
* For achieving the highest quality labels, GPT-4 is the best choice among out of the box LLMs (88.4% agreement with ground truth, compared to 86% for skilled human annotators). 
* For achieving the best tradeoff between label quality and cost, GPT-3.5-turbo, PaLM-2 and open source models like FLAN-T5-XXL are compelling.
* Confidence based thresholding can be a very effective way to mitigate impact of hallucinations and ensure high label quality.
