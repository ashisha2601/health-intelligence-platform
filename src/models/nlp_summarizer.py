
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
        
    def _load_model(self):
        """Lazy load the model pipeline."""
        if self.summarizer is None:
            logger.info(f"Loading summarization model: {self.model_name}...")
            device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
            
            # Check for MPS (Apple Silicon) specifically
            if torch.backends.mps.is_available():
                device = "mps"
            
            try:
                self.summarizer = pipeline("summarization", model=self.model_name, device=device)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def summarize(self, text, max_length=150, min_length=40):
        """
        Summarize a single clinical note.
        """
        self._load_model()
        
        if not text or len(text.strip()) == 0:
            return ""
            
        try:
            # Estimate token count roughly (words * 1.3)
            # Adjust max_length to be reasonable for the input size to avoid warnings
            estimated_tokens = int(len(text.split()) * 1.5)
            dynamic_max = min(max_length, max(estimated_tokens, min_length + 10))
            
            # Ensure min_length is valid
            dynamic_min = min(min_length, dynamic_max - 1)
            
            summary = self.summarizer(text, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
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
