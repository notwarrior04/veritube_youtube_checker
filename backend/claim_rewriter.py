import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

from transformers import pipeline

# Load once globally
rewriter = pipeline("text2text-generation", model="google/flan-t5-base")

def rewrite_claim(claim):
    prompt = f"Rewrite the following claim to make it clearer and more specific: {claim}"
    try:
        output = rewriter(prompt, max_length=64, max_new_tokens=None, do_sample=False)
        return output[0]["generated_text"].strip()
    except Exception as e:
        print(f"⚠️ Claim rewriting failed: {e}")
        return claim  # fallback to original if error
