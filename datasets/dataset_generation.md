## Findings from playing around with dataset synthesis

### 1. reasoning models are bad at generating data

They can't just directly output a response, which makes it so synthesizing lots of examples is hard because you would need to perform some sort of extraction on output. Also, expensive.

### llama-3.3-70b-versatile has very bare bones outputs based on prompt from gpt-4o
**Ex. \<diagnostic_test\> \<patient\> \<misinterpretation\>**  
 HISTORY OF PRESENT ILLNESS  
The patient is a 45-year-old male who presents with symptoms that began two weeks ago, initially misinterpreting his electrocardiogram as a normal reading, when in fact it showed signs of cardiac ischemia, the patient himself did not understand the results of the electrocardiogram. 

PAST MEDICAL HISTORY  
The patient has a history of hypertension, hyperlipidemia, and a family history of coronary artery disease, all of which were considered when ordering the electrocardiogram.

ASSESSMENT  
The patient underwent an electrocardiogram diagnostic test, and after reevaluation, the results indicated cardiac ischemia in the patient, which was initially a misinterpretation by the patient of the electrocardiogram results.

like llama-3.2-11b-vision-preview outputs on new prompt

like llama3-70b-8192, but output gets cut off with 700 tokens? doesnt really matter cuz groq is free

like llama-3.3-70b-versatile

like  llama-3.1-8b-instant

like llama-3.2-3b-preview

like  llama-3.2-90b-vision-preview

going to try out llama-3.3-70b-versatile with negative examples

### I have found that the results while good for 1 run tend to follow a very similar story, history of smoking, elderly patients. Probably due to data distribution being trained on (may be reflective of wider data distribution), but still want more variability so going to try switching the prompt around more.

Initial go at making it diverse made it way too diverse, and also just people in their young 30s?

Increased temperature for more variability, liking what I am getting at 2. Going to stop here for now and generate another dataset, but this could def use more work.