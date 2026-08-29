import os 
import urllib.request # Mini-dataset of open-source AI papers 

pdf_urls = { "Attention_Is_All_You_Need.pdf": "https://arxiv.org/pdf/1706.03762.pdf", 
            "LLaMA.pdf": "https://arxiv.org/pdf/2302.13971.pdf", 
            "LoRA.pdf": "https://arxiv.org/pdf/2106.09685.pdf", 
            "RAG.pdf": "https://arxiv.org/pdf/2005.11401.pdf", 
            "ChunkNorris.pdf": "https://arxiv.org/pdf/2602.00010.pdf" } 


os.makedirs("./data", exist_ok=True) 
for filename, url in pdf_urls.items(): 
    filepath = os.path.join("./data", filename) 
    if not os.path.exists(filepath): 
        print(f"Downloading {filename}...") 
        urllib.request.urlretrieve(url, filepath) 
        print("Dataset ready in the ./data folder!") 