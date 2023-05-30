# Modules

On this page, we will talk about the different pages that exist in AutoLabel. We will first discuss the overview of a module and then go into the different subheadings, expanding and giving some examples for each.

## Prompts

Writing prompts is a crucial aspect of training language models for specific tasks. In this tutorial, we will explore the five essential parts of a prompt: the prefix prompt, task prompt, output prompt, seed examples, and current example. Understanding and constructing these components effectively can help guide the model's behavior and generate accurate and contextually appropriate responses. Let's delve into each part in detail.

### Prefix Prompt
The prefix prompt is the initial line of the prompt, which sets the domain and provides task-independent information to the model. It helps the model understand the specific area or expertise it should embody while generating responses. For example, if the prefix prompt indicates a medical domain, the model will focus on generating responses that align with medical knowledge and terminology.  
Example:  
[Medical] In this prompt, the model should provide expert advice on diagnosing and treating common ailments.

### Task Prompt
The task prompt explains the objective or task the model needs to accomplish. It describes the specific instructions or guidelines for completing the task. This section is crucial for clearly conveying the desired output from the model.  
Example:  
You are a medical expert. Given a patient's symptoms and medical history, provide a diagnosis and recommend appropriate treatment options.

### Output Prompt
The output prompt informs the model about the expected answer format or structure. It defines the specific format in which the model should provide the answer. This step ensures consistency and enables easier processing of the model's responses.
Example:  
Provide the diagnosis and treatment recommendations in JSON format, with the following keys: "diagnosis" and "treatment." The value for each key should be a string representing the diagnosis and treatment, respectively.

### Seed Examples
Seed examples play a vital role in training the model by providing real-world examples from the task distribution. These examples help the model grasp the nature of the task, understand the expected outputs, and align its behavior accordingly. It is crucial to provide meaningful and diverse seed examples to facilitate accurate responses.
Example:  
Seed Examples:  

Patient: Fever, sore throat, and fatigue. Medical History: None.  
Diagnosis: "Common cold"  
Treatment: "Rest, plenty of fluids, and over-the-counter cold medication."  
Patient: Persistent cough, shortness of breath, and wheezing. Medical History: Asthma.  
Diagnosis: "Asthma exacerbation"  
Treatment: "Inhaled bronchodilators and corticosteroids as prescribed."

### Current Example
The current example is the specific instance for which you seek the model's response. It provides the exact answer or label you want the model to assign to this particular example.  
Example:  
Current Example:  
Patient: Severe headache, visual disturbances, and nausea. Medical History: None.  
Desired Diagnosis: "Migraine"  
Desired Treatment: "Prescribed pain-relief medication and lifestyle modifications."  

## Configs

There are 3 modules required by every labeling run -
1. A task
2. An LLM
3. A dataset

All 3 of these modules can be instantiated with configs. A config can be passed in as a dictionary or as the path to a json file. The config consists of different keys and the following section will list out each key along with the property of the module that it affects.

### Config

The Config class is used to parse, validate, and store information about the labeling task being performed.

::: src.autolabel.configs.config.AutolabelConfig
    rendering:
        show_root_full_path: no
        heading_level: 4

## Tasks

### Classification
### Question Answering
### Entity matching
### Named Entity Recognition

## LLMs

There are a few LLMs that are available for labeling through the AutoLabel library. Some of these are supported behind an API (OpenAI, Anthropic, Refuel) and some run locally (Huggingface pipeline).  
  
These LLMs are classified by the LLM providers, these are the organizations which provide access to the LLMs. The second parameter, is the model_name. The same provider can have multiple models exposed, for eg. OpenAI exposes text-davinci-003 and gpt-3.5-turbo.

### Openai
The provider_name here is 'openai'. The models supported by openai are text-davinci-003, gpt-3.5-turbo and gpt4. Every model has a different pricing strategy and could be useful while choosing the best model for your use case.  

A few parameters that can be passed in for openai models:
max_tokens (int) - The maximum tokens to sample from the model
temperature (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.

### Anthropic
The provider_name here is 'anthropic'. The models supported are claude-v1, claude-instant-v1.  

A few parameters that can be passed in for anthopic models:
max_tokens_to_sample (int) - The maximum tokens to sample from the model
temperature (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.

### Huggingface Pipeline
The provider_name here is huggingface_pipeline. Any model available on the huggingface hub can be passed in as the model name. This runs the model locally on a GPU if that is available. Some models may be too big to fit on a GPU available locally to users, and so you can specify quantization strategy which makes the model smaller in terms of memory.

A few parameters that can be passed in for huggingface_pipeline models:
max_new_tokens (int) - The maximum tokens to sample from the model
temperature (float) - A float b/w 0 and 1 which indicates the diversity you want in the output. 0 uses greedy sampling.
quantize (int) - The model quantization to use. 32 bit by default, but we also support 16 bit and 8 bit support for models which have been hosted on huggingface.

### Refuel model
