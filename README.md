# RAG-assistant for research papers 



### How to use : 
installing the depndencies : 
minimal : 
heavy (for experiments) : 

> folder descriotion : 
experiments/ : contianing my expertiment records on a single paper.
pipeline/ pridcuton grade-ready script. cleaned and tested to the best.


usage : 
run : 



### Parsing :
pdfPlumber : Trash, failst to extract text, slow compared to PyMuPdf with far better results.
PyMuPDF    : decent results for text with fast execution time
Unstructured : very heavy in dependences, very slow, and deepLearning based : think of having enoughmemory/ressource for it.
docling : 44.87

> Images Handling : 


### Chunking :
To the best of my knowledge, best practises for RAG advice to use chunks around 512 token. ( ressource : https://www.premai.io/blog/rag-chunking-strategies-the-2026-benchmark-guide/)

applying the general rule $ N_{tokens} \approx N_{words} \times 1.5 $. 512 tokens usually fit 340 word.
  


My chuking strategy then is the following : 
- for each chunk, keep a header containing the name of the paper it came from, and the name the section (abstract/Results...) --> explanations : for me this will give tremndous help for the BM25 Rietriving to answer question that are very domain specefic (metrics names, paper titles...) and that are fuzzy for the dense retrieving, like the user's specefic questions : 'Based on research papers abstract, write for me an abstract for this case' or 'what is the learning rate used to train the model in the paper attention is all you need".

- since we already have a prior knowledge that meaning completely shifts from section to another, semantics are inherently preserved through header, hence  it is needless to go for complex semantic-chunking strategies.

Additionly, and that by practise authors follow some practises to keep paragraph shorts, (confirmed by statistics I found here https://quantifyinghealth.com/paragraph-length-in-research-papers/?utm_source=chatgpt.com, keeping on average a count of 125 words, up to 286 words to the 95% percentile), my strategy is the following : for every new section, start a new chunk --> keep stuffing paragraph by paragraph --> if the number of tolens exceed 512, start a new chunk, with the same header of before. repeat untill a new section is found.

- preprocessing strategies using Regex! :
filter the references [1,2], since they boost only the reader experience, but irrelevant to the RAG task. 






# Personal feedback and what I have learnt NEW from this exervise :

-  "lost in the middle" phenomenon : when LLMs process long texts, they pay attention only to the start and the end, ignoring important facts hidden in the middle. --> result :  having a huge-context-window LLM doens't mean we should stuff the input with much contexts  : best practice : 5 chunks 512 token each.