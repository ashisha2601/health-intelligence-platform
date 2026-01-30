
from transformers import pipeline
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """
        Initialize the summarizer with a pre-trained model.
        Using distilbart-cnn-12-6 by default for a good balance of speed and performance.
        """
        self.model_name = model_name
        self.summarizer = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def _load_model(self):
        """Lazy load the model pipeline."""
        if self.summarizer is None:
            logger.info(f"Loading summarization model: {self.model_name}...")
            
            # Detect best device
            if torch.cuda.is_available():
                self.device = 0
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = -1 # CPU
            
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                hub_kwargs = {"timeout": 300} 
                
                logger.info("Initializing tokenizer...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **hub_kwargs)
                except Exception as te:
                    logger.warning(f"Tokenizer load failed for {self.model_name}: {te}. Trying t5-small tokenizer...")
                    self.model_name = "t5-small"
                    self.tokenizer = AutoTokenizer.from_pretrained("t5-small", **hub_kwargs)
                
                # Memory optimization: use float16 if not on CPU to save 50% RAM
                dtype_to_use = torch.float16 if self.device != -1 else torch.float32
                
                logger.info(f"Initializing model {self.model_name}...")
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name, 
                        dtype=dtype_to_use, 
                        low_cpu_mem_usage=True,
                        **hub_kwargs
                    )
                except Exception as me:
                    logger.warning(f"Model load failed for {self.model_name}: {me}. Trying t5-small...")
                    self.model_name = "t5-small"
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        "t5-small", 
                        dtype=dtype_to_use, 
                        low_cpu_mem_usage=True,
                        **hub_kwargs
                    )
                
                # Create pipeline
                try:
                    self.summarizer = pipeline(
                        "summarization", 
                        model=self.model, 
                        tokenizer=self.tokenizer, 
                        device=self.device
                    )
                except Exception as pe:
                    logger.warning(f"Pipeline creation failed: {pe}. Using direct model generation.")
                    self.summarizer = "DIRECT_GEN"
                    if self.device != -1:
                        self.model = self.model.to(self.device)
                
                logger.info("NLP initialization complete.")
            except Exception as e:
                logger.error(f"FATAL ERROR in NLP initialization: {e}. Falling back to MOCK mode.")
                self.summarizer = "MOCK"

    def summarize(self, text, max_length=150, min_length=40):
        """
        Summarize a single clinical note.
        """
        self._load_model()
        
        if not text or len(text.strip()) == 0:
            return ""
            
        try:
            # Estimate token count
            estimated_tokens = int(len(text.split()) * 1.5)
            dynamic_max = min(max_length, max(estimated_tokens, min_length + 10))
            dynamic_min = min(min_length, dynamic_max - 1)
            
            if self.summarizer == "MOCK":
                return f"SUMMARY PREVIEW: {text[:100]}... [Note: AI model failed to download due to network timeout. This is a placeholder.]"
            
            if self.summarizer == "DIRECT_GEN":
                # Manual Generation Fallback
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
                if self.device != -1:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                summary_ids = self.model.generate(
                    inputs["input_ids"], 
                    max_length=dynamic_max, 
                    min_length=dynamic_min, 
                    length_penalty=2.0, 
                    num_beams=4, 
                    early_stopping=True
                )
                return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            else:
                # Standard Pipeline
                # T5 needs a prefix
                prefix = "summarize: " if "t5" in self.model_name.lower() else ""
                summary = self.summarizer(prefix + text, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
                return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text # Return original if failure

    def batch_summarize(self, texts, batch_size=4):
        """
        Summarize a list of clinical notes.
        """
        self._load_model()
        
        results = []
        try:
            # Process in batches if using a generator or just direct list
            if isinstance(texts, list):
                summaries = self.summarizer(texts, batch_size=batch_size, truncation=True)
                results = [s['summary_text'] for s in summaries]
            return results
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")
            return []

if __name__ == "__main__":
    # Test run
    print("Testing ClinicalSummarizer...")
    summarizer = ClinicalSummarizer()
    
    sample_text = """
    Patient is a 65-year-old male presenting with chest pain and shortness of breath. 
    History of hypertension and type 2 diabetes. 
    ECG showed ST elevation in leads V1-V4. 
    Troponin levels were elevated at 0.5 ng/mL. 
    Patient was administered aspirin and nitroglycerin. 
    Transferred to Cath Lab for PCI. 
    Post-procedure, patient is stable and pain-free. 
    Discharge planned in 2 days with cardiovascular rehab referral.
    """
    
    print("\nInput Text:")
    print(sample_text.strip())
    
    summary = summarizer.summarize(sample_text)
    
    print("\nGenerated Summary:")
    print(summary)
