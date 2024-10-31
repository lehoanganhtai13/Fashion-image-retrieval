from typing import List, Dict
from threading import Thread

import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, pipeline
from FlagEmbedding import FlagLLMReranker
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TextIteratorStreamer

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    },
    "icl": {
        "query": "Convert this example into vector to look for useful examples: ",
        "key": "Convert this example into vector for retrieval: ",
    },
    "chat": {
        "query": "Embed this dialogue to find useful historical dialogues: ",
        "key": "Embed this historical dialogue for retrieval: ",
    },
    "lrlm": {
        "query": "Embed this text chunk for finding useful historical chunks: ",
        "key": "Embed this historical text chunk for retrieval: ",
    },
    "tool": {
        "query": "Transform this user request for fetching helpful tool descriptions: ",
        "key": "Transform this tool description for retrieval: "
    },
    "convsearch": {
        "query": "Encode this query and context for searching relevant passages: ",
        "key": "Encode this passage for retrieval: ",
    },
}

# See: https://huggingface.co/openai/clip-vit-base-patch32 and https://huggingface.co/patrickjohncyh/fashion-clip
class CLIPEncoder():
    """CLIP Encoder class to encode text and image data."""
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id, clean_up_tokenization_spaces=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=True)
    
    def get_image_embeddings(self, images):
        """Get the embedding of a single image."""
        image_inputs = self.processor(
            text = None,
            images = images, 
            return_tensors="pt",
            padding=True
        )["pixel_values"].to(self.device)
        embeddings = self.model.get_image_features(image_inputs)

        # convert the embeddings to list
        embeddings_as_list = embeddings.cpu().detach().numpy().tolist()
        
        # Clean up variables to free up memory in the GPU
        del image_inputs
        del embeddings
        torch.cuda.empty_cache()

        return embeddings_as_list

    def get_text_embeddings(self, texts):
        """Get the embeddings of a batch of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        print(f"Tokenized input shape: {inputs['input_ids'].shape}")
        text_embeddings = self.model.get_text_features(**inputs)
        print(f"Text embeddings shape: {text_embeddings.shape}")

        # convert the embeddings to list 
        embeddings_as_list = text_embeddings.cpu().detach().numpy().tolist()

        # Clean up variables to free up memory in the GPU
        del inputs
        del text_embeddings
        torch.cuda.empty_cache()
            
        return embeddings_as_list

# See: https://arxiv.org/abs/2407.01219 and https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/llm_embedder
class Embedder():
    def __init__(self, model_id: str = "BAAI/llm-embedder", cache_dir: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            cache_folder=cache_dir
        )
        self.instruction = INSTRUCTIONS["qa"]

    def embed_query(self, queries: List[str]) -> torch.Tensor:
        queries = [self.instruction["query"] + query for query in queries]
        query_embeddings = self.model.encode(queries)
        embeddings_list = query_embeddings.tolist()
        del query_embeddings, queries
        torch.cuda.empty_cache()
        return embeddings_list

    def embed_doc(self, docs: List[str]) -> torch.Tensor:
        docs = [self.instruction["key"] + doc for doc in docs]
        doc_embeddings = self.model.encode(docs)
        embeddings_list = doc_embeddings.tolist()
        del doc_embeddings, docs
        torch.cuda.empty_cache()
        return embeddings_list

# See: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
class LLM():
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct", HF_token: str = "", cache_dir: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        if self.device == "cpu":
            torch_type = torch.bfloat16
            model_id = "meta-llama/Llama-3.2-3B-Instruct" # This will take about 3 minutes to generate using CPU
        else:
            torch_type = torch.float16
        #     model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.model = pipeline(
            "text-generation",
            model=model_id,
            device=self.device,
            model_kwargs={"torch_dtype": torch_type, "cache_dir": cache_dir},
            token=HF_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        return self.model(
            messages, max_new_tokens=max_new_tokens, 
            pad_token_id=self.model.tokenizer.eos_token_id
        )[0]["generated_text"][-1]

    def streaming(self, messages: List[Dict[str, str]], max_new_tokens: int):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            text_inputs=messages, 
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        thread = Thread(target=self.model, kwargs=generation_kwargs)
        thread.start()
        return streamer

# See: https://huggingface.co/BAAI/bge-reranker-v2-gemma
class ReRanker():
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-gemma", cache_dir: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        use_fp16 = True if self.device != "cpu" else False
        self.model = FlagLLMReranker(
            model_id,
            use_fp16=use_fp16, # Setting use_fp16 to True speeds up computation with a slight performance degradation
            device=self.device,
            cache_dir=cache_dir
        )

    def rerank(self, pairs: List[List[str]]) -> List[int]:
        scores = self.model.compute_score(pairs)
        del pairs
        torch.cuda.empty_cache()
        return scores
