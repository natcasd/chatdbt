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

# Comments on generating s_regex3, patient_records3, patient_records3_nl_query, patient_records4, patient_records4_nl_query
After reviewing our dataset patient_records2.json and patient_records2_nl_query.json we had three main observations:

1. The model we used to generate the data was llama-3.3-70b-versatile, which also performed exceptionally well on evaluation. 
2. Some of the patient records directly state the words of the semantic symbols instead of incorporating them into the story of the patient record.
3. A lot of the records are about patients who smoked.

Our solution to address these problems were:

1. Use GPT-4o to generate a new dataset which would introduce variation in the provider of the models and potentially the data the model was trained on. To further address this we will evaluate using three different models which should mitigate the benfits, if any, of traning and evaluting with the same model.
2. After further analysis we believe this is not a severe issue as the nature of the queries will vary and sometimes the query could be located consecutively in a patient record or not.
3. After furhter analysis we attributed the frequency of smoking in patients to the medical process of asking patients about their life habits. Questions about drinking and smoking are routinely asked which
may seem like it is common but is rather a reflection of routine questions.

We then generated s_regex3 using GPT-4o, a list of semantic patterns that would guide our patient record creation. In our experimentation we used different prompts as well as temperature values. We settled on a prompt that directed the generation enough with specific examples, but still left room for creativity and improvisation. We noticed that temperature at 1 yielded more reliable results. 

Using s_regex3 we tested the generation of negative and positive examples for a few instances to examine the quality. The original prompt used to generate patient_records2 was effective so we decided to keep it. We determined that generating using a different model (GPT-4o instead of llama 3.3 70b) and having different semantic regexes would be sufficient variation.

Additionally we create patient_records4, which is the combination of patient_records2 and patient_records3. We also shuffled the data so that a positive instance would not be directly followed by a negative one. We hope this will introduce more complexity to the problem and make the evaluation a more accurate representation of the viability of approach 1. 