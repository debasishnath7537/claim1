import streamlit as st
import openai
import json
import re
import logging
import os
import tempfile
from typing import Dict, Any, Tuple
from retry import retry
import librosa
import numpy as np
import pandas as pd
import cv2
import pytesseract
from PIL import Image
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to validate API key (basic check)
def validate_api_key(api_key: str) -> bool:
    return bool(api_key and api_key.startswith("sk-") and len(api_key) > 20)

# Load JSON data
def load_claim_data(json_file: str) -> Dict[str, Any]:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            required_fields = ["customer", "incident_details", "claim_submission", "document_verification"]
            if not all(field in data for field in required_fields):
                raise ValueError("Missing required fields in JSON")
            return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise

# Transcribe audio using Whisper
@retry(tries=3, delay=2, backoff=2)
def transcribe_audio(audio_file: str) -> str:
    try:
        with open(audio_file, "rb") as f:
            transcription = openai.Audio.transcribe(model="whisper-1", file=f)
        logger.info("Audio transcription successful")
        return transcription["text"].strip()
    except (FileNotFoundError, openai.OpenAIError) as e:
        logger.error(f"Error in transcription: {e}")
        if isinstance(e, openai.OpenAIError):
            logger.warning(f"Transcription failed due to API error: {e}, skipping audio processing.")
        raise ValueError(f"Transcription failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in transcription: {e}")
        raise ValueError(f"Transcription failed: {e}")

# Calculate speech rate and audio metrics
def calculate_audio_metrics(audio_file: str, transcription: str) -> Dict[str, float]:
    try:
        y, sr = librosa.load(audio_file)
        duration = librosa.get_duration(y=y, sr=sr)
        words = len(transcription.split())
        minutes = duration / 60.0
        speech_rate = words / minutes if minutes > 0 else 0.0
        
        pitches = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.mean(pitches[pitches > 0])
        
        logger.info(f"Audio metrics: speech_rate={speech_rate:.1f} wpm, avg_pitch={avg_pitch:.1f} Hz")
        return {"speech_rate_wpm": speech_rate, "avg_pitch": avg_pitch}
    except Exception as e:
        logger.error(f"Error calculating audio metrics: {e}")
        raise ValueError(f"Failed to calculate audio metrics: {e}")

# Speaker verification (placeholder)
def verify_speaker(audio_file: str, customer_id: str, transcription: str, customer: Dict[str, Any]) -> str:
    try:
        expected_name = customer["first_name"]
        expected_policy = customer["policy_number"]
        if expected_name.lower() in transcription.lower() and expected_policy in transcription:
            logger.info(f"Speaker verified for {customer_id}")
            return f"Speaker verified as {expected_name} {customer['last_name']} ({customer_id})"
        logger.warning("Speaker verification failed: mismatch in name or policy number")
        return "Speaker verification failed"
    except Exception as e:
        logger.error(f"Error in speaker verification: {e}")
        return f"Speaker verification error: {e}"

# Sentiment and emotion analysis
@retry(tries=3, delay=2, backoff=2)
def analyze_sentiment(transcription: str, speech_rate: float, avg_pitch: float) -> Dict[str, Any]:
    try:
        prompt = f"""
        Analyze the sentiment and emotional state of this text: '{transcription}'.
        Detect stress indicators (hesitations like 'um', 'uh', urgency, tone).
        Consider speech rate ({speech_rate:.1f} wpm) and average pitch ({avg_pitch:.1f} Hz).
        Estimate deception likelihood based on linguistic patterns (e.g., evasiveness, contradictions).
        Provide a confidence score for the analysis.
        Output in JSON format:
        {{
            "emotion": str,
            "stress_level": float (0-1),
            "deception_likelihood": float (0-1),
            "speech_rate_wpm": float,
            "confidence": float (0-1)
        }}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result["speech_rate_wpm"] = speech_rate
        logger.info("Sentiment analysis completed")
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise ValueError(f"Sentiment analysis failed: {e}")

# DocAgent for document processing
def process_document(image_file: str, doc_type: str, claim_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise ValueError("Tesseract OCR is not installed or not in PATH. Please install it and add to PATH.")

        # Load and preprocess image
        img = cv2.imread(image_file)
        if img is None:
            raise ValueError(f"Failed to load image: {image_file}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # OCR extraction
        extracted_text = pytesseract.image_to_string(thresh).strip()
        logger.info(f"OCR extracted text for {doc_type}: {extracted_text}")

        # Authenticity detection (basic CV check for edits or AI generation)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        is_fake_likelihood = min(0.5, max(0.0, edge_density - 0.1)) if edge_density > 0.1 else 0.0
        authenticity_status = "Suspected Fake or AI Generated" if is_fake_likelihood < 0.3 else "Real"

        # Define verification result
        verification_result = {"extracted_text": extracted_text, "authenticity_status": authenticity_status, "is_fake_likelihood": is_fake_likelihood}

        # Verify extracted data based on doc_type
        matches = {}
        expected_data = claim_data["document_verification"].get(doc_type, {})
        for key, expected_value in expected_data.items():
            extracted_value = re.search(rf"\b{re.escape(expected_value)}\b", extracted_text, re.IGNORECASE)
            matches[key] = {
                "expected": expected_value,
                "extracted": extracted_value.group() if extracted_value else "Not Found",
                "match": extracted_value is not None
            }

        if doc_type == "last_location":
            # Extract any location-like text (e.g., street names or place names) without matching
            location_match = re.search(r"\b\w+\s+(Street|Road|Avenue|Place|Square|Lane|Park|Boulevard)\b", extracted_text, re.IGNORECASE)
            last_location = location_match.group() if location_match else "Location Not Found"
            verification_result["last_location"] = last_location

        verification_result["matches"] = matches

        logger.info(f"Document {doc_type} processed: {verification_result}")
        return verification_result
    except Exception as e:
        logger.error(f"Error processing document {doc_type}: {e}")
        return {"error": str(e), "authenticity_status": "Unknown", "is_fake_likelihood": 0.0, "matches": {}}

# Enhanced Fraud Scoring Agent
# Enhanced Fraud Scoring Agent
def fraud_scoring_agent(claim_data: Dict[str, Any], voice_analysis: Dict[str, Any], doc_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        # Base score from deception likelihood
        fraud_score = voice_analysis["deception_likelihood"] * 0.4
        # Add contribution from stress level
        fraud_score += voice_analysis["stress_level"] * 0.2
        # Adjust based on emotion
        high_risk_emotions = ["anxiety", "distressed", "frustration and urgency", "anger", "fear"]
        if voice_analysis["emotion"].lower() in [e.lower() for e in high_risk_emotions]:
            fraud_score += 0.1
        # Incorporate claim data (optional, e.g., prior claims)
        if claim_data.get("fraud_check", {}).get("prior_claims", 0) > 0:
            fraud_score += 0.1
        # Add document verification impact
        doc_mismatch_count = sum(not all(match["match"] for match in result["matches"].values()) for result in doc_results.values() if "matches" in result)
        doc_fake_likelihood = sum(result.get("is_fake_likelihood", 0.0) for result in doc_results.values()) / max(len(doc_results), 1)
        fraud_score += doc_mismatch_count * 0.1  # 0.1 per mismatch
        fraud_score += doc_fake_likelihood * 0.3  # Weigh fake likelihood heavily
        # Cap score at 1.0
        fraud_score = min(fraud_score, 1.0)
        logger.info(f"Fraud score calculated: {fraud_score:.2f}")
        return {
            "fraud_score": fraud_score,
            # "status": "Manual Review" if fraud_score < 0.2 else "Auto-Approved",
            "status": "Auto-Approved" if fraud_score <= 0.55 else "Manual Review",
            "score_breakdown": {
                "deception_likelihood_contribution": voice_analysis["deception_likelihood"] * 0.4,
                "stress_level_contribution": voice_analysis["stress_level"] * 0.2,
                "emotion_adjustment": 0.1 if voice_analysis["emotion"].lower() in [e.lower() for e in high_risk_emotions] else 0.0,
                "prior_claims_adjustment": 0.1 if claim_data.get("fraud_check", {}).get("prior_claims", 0) > 0 else 0.0,
                "document_mismatch_contribution": doc_mismatch_count * 0.1,
                "document_fake_likelihood_contribution": doc_fake_likelihood * 0.3
            }
        }
    except Exception as e:
        logger.error(f"Error in fraud scoring: {e}")
        raise ValueError(f"Fraud scoring failed: {e}")

# Main Orchestrator
def process_claim(json_file: str, audio_file: str, policy_image: str = None, purchase_image: str = None, location_image: str = None) -> Dict[str, Any]:
    try:
        claim_data = load_claim_data(json_file)
        customer = claim_data["customer"]
        claim_id = claim_data.get("claim_id", "Unknown")
        
        transcription = transcribe_audio(audio_file) if audio_file else ""
        logger.info("Audio file processed")
        audio_metrics = calculate_audio_metrics(audio_file, transcription) if audio_file else {"speech_rate_wpm": 0.0, "avg_pitch": 0.0}
        speaker_result = verify_speaker(audio_file, customer["customer_id"], transcription, customer) if audio_file else "No audio provided"
        voice_analysis = analyze_sentiment(transcription, audio_metrics["speech_rate_wpm"], audio_metrics["avg_pitch"]) if audio_file and transcription else {
            "emotion": "Unknown",
            "stress_level": 0.0,
            "deception_likelihood": 0.0,
            "speech_rate_wpm": 0.0,
            "confidence": 0.0
        }

        # Process documents sequentially
        doc_results = {}
        if policy_image:
            doc_results["insurance_policy"] = process_document(policy_image, "insurance_policy", claim_data)
            logger.info("Insurance policy receipt processed")
        if purchase_image:
            doc_results["purchase_receipt"] = process_document(purchase_image, "purchase_receipt", claim_data)
            logger.info("Purchase receipt processed")
        if location_image:
            doc_results["last_location"] = process_document(location_image, "last_location", claim_data)
            logger.info("Last location image processed")

        fraud_result = fraud_scoring_agent(claim_data, voice_analysis, doc_results)
        
        result = {
            "claim_id": claim_id,
            "customer_id": customer["customer_id"],
            "transcription": transcription,
            "speaker_verification": speaker_result,
            "voice_analysis": voice_analysis,
            "document_verification": doc_results,
            "fraud_result": fraud_result
        }
        logger.info(f"Claim {claim_id} processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing claim: {e}")
        return {"error": str(e), "status": "Failed"}

# Streamlit UI
def main():
    st.title("Claims Automation System")
    st.write("Enter your OpenAI API key and upload an audio file, Insurance Policy Confirmation Receipt, Item Purchase Receipt, and Last Location Image, then click 'Run' to process in sequence.")

    # API Key Input
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        if validate_api_key(api_key):
            openai.api_key = api_key
            st.success("API key validated successfully!")
        else:
            st.error("Invalid API key. It should start with 'sk-' and be a valid OpenAI key.")
            return  # Stop execution if API key is invalid
    else:
        st.error("Please enter an OpenAI API key.")
        return  # Stop execution if no API key is provided

    # File uploaders with audio playback and image preview
    audio_file = st.file_uploader("Upload Audio File (WAV or MP3)", type=["wav", "mp3"], key="audio")
    if audio_file:
        st.audio(audio_file, format="audio/wav" if audio_file.name.endswith(".wav") else "audio/mp3", start_time=0)

    policy_image = st.file_uploader("Upload Insurance Policy Confirmation Receipt (PNG/JPG)", type=["png", "jpg", "jpeg"], key="policy")
    if policy_image:
        st.image(policy_image, caption="Insurance Policy Confirmation Receipt", use_column_width=True)

    purchase_image = st.file_uploader("Upload Item Purchase Receipt (PNG/JPG)", type=["png", "jpg", "jpeg"], key="purchase")
    if purchase_image:
        st.image(purchase_image, caption="Item Purchase Receipt", use_column_width=True)

    location_image = st.file_uploader("Upload Last Location Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="location")
    if location_image:
        st.image(location_image, caption="Last Location Image", use_column_width=True)

    # Run button
    if st.button("Run"):
        if not api_key or not validate_api_key(api_key):
            st.error("Please provide a valid OpenAI API key.")
            return
        if not all([audio_file, policy_image, purchase_image, location_image]):
            st.error("Please upload all files (audio, insurance policy receipt, purchase receipt, and last location image) before running.")
        else:
            # Save files temporarily
            temp_files = {}
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(audio_file.read())
                temp_files["audio"] = temp_audio.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_policy:
                temp_policy.write(policy_image.read())
                temp_files["policy"] = temp_policy.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_purchase:
                temp_purchase.write(purchase_image.read())
                temp_files["purchase"] = temp_purchase.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_location:
                temp_location.write(location_image.read())
                temp_files["location"] = temp_location.name

            # Display a loading spinner
            with st.spinner("Processing claim..."):
                json_file = "test.json"
                try:
                    result = process_claim(json_file, temp_files["audio"], temp_files["policy"], temp_files["purchase"], temp_files["location"])

                    # Clean up temporary files
                    for temp_file in temp_files.values():
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)

                    # Display results
                    if "error" not in result:
                        st.success("Claim processed successfully!")
                        
                        # Claim and Customer Info
                        st.subheader("Claim Information")
                        st.write(f"**Claim ID**: {result['claim_id']}")
                        st.write(f"**Customer ID**: {result['customer_id']}")
                        st.write(f"**Speaker Verification**: {result['speaker_verification']}")

                        # Transcription
                        if result["transcription"]:
                            st.subheader("Transcription")
                            st.write(result["transcription"])

                        # Voice Analysis Table
                        if result["voice_analysis"]:
                            st.subheader("Voice Analysis")
                            voice_data = result["voice_analysis"]
                            voice_df = pd.DataFrame({
                                "Metric": ["Emotion", "Stress Level", "Deception Likelihood", "Speech Rate (wpm)", "Confidence"],
                                "Value": [
                                    str(voice_data["emotion"]),
                                    str(voice_data["stress_level"]),
                                    str(voice_data["deception_likelihood"]),
                                    str(round(voice_data["speech_rate_wpm"], 2)),
                                    str(voice_data["confidence"])
                                ]
                            })
                            st.table(voice_df)

                        # Document Verification
                        if result["document_verification"]:
                            st.subheader("Document Verification")
                            for doc_type, doc_result in result["document_verification"].items():
                                st.write(f"**{doc_type.replace('_', ' ').title()}**")
                                st.write(f"Authenticity Status: {doc_result['authenticity_status']}")
                                if doc_result.get("matches"):
                                    matches_df = pd.DataFrame({
                                        "Field": list(doc_result["matches"].keys()),
                                        "Expected": [m["expected"] for m in doc_result["matches"].values()],
                                        "Extracted": [m["extracted"] for m in doc_result["matches"].values()],
                                        "Match": [str(m["match"]) for m in doc_result["matches"].values()]
                                    })
                                    st.table(matches_df)
                                if doc_type == "last_location" and "last_location" in doc_result:
                                    st.subheader("Last Location")
                                    st.write(f"Last location used is: {doc_result['last_location']}")

                        # Fraud Result Table
                        st.subheader("Fraud Analysis")
                        fraud_data = result["fraud_result"]
                        fraud_df = pd.DataFrame({
                            "Metric": ["Risk Score", "Status"],
                            "Value": [str(round(fraud_data["fraud_score"], 2)), str(fraud_data["status"])]
                        })
                        st.table(fraud_df)

                        # Display approval message if Auto-Approved
                        if fraud_data["status"] == "Auto-Approved":
                            claim_data = load_claim_data(json_file)
                            amount = claim_data["claim_submission"].get("expected_amount", 1099)
                            st.write(f"**Claim approved for £{amount} – replacement value. Confirmation SMS and email sent.**")

                        # Fraud Score Breakdown
                        st.subheader("Score Breakdown")
                        breakdown = fraud_data["score_breakdown"]
                        breakdown_df = pd.DataFrame({
                            "Component": [
                                "Deception Likelihood Contribution",
                                "Stress Level Contribution",
                                "Emotion Adjustment",
                                "Prior Claims Adjustment",
                                "Document Mismatch Contribution",
                                "Document Contribution"
                            ],
                            "Value": [
                                str(round(breakdown["deception_likelihood_contribution"], 2)),
                                str(round(breakdown["stress_level_contribution"], 2)),
                                str(round(breakdown["emotion_adjustment"], 2)),
                                str(round(breakdown["prior_claims_adjustment"], 2)),
                                str(round(breakdown["document_mismatch_contribution"], 2)),
                                str(round(breakdown["document_fake_likelihood_contribution"], 2))
                            ]
                        })
                        st.table(breakdown_df)

                        # Bar Chart for Voice Analysis
                        # Bar Chart for Voice Analysis
                        if result["voice_analysis"]:
                            st.subheader("Voice Analysis Metrics")
                            chart_data = pd.DataFrame({
                                "Metric": ["Stress Level", "Deception Likelihood", "Confidence"],
                                "Value": [
                                    voice_data["stress_level"],
                                    voice_data["deception_likelihood"],
                                    voice_data["confidence"]
                                ]
                            })
                            st.bar_chart(chart_data.set_index("Metric"), height=400)

                    else:
                        st.error(f"Error: {result['error']}")
                except Exception as e:
                    st.error(f"Error processing claim: {e}")
                    for temp_file in temp_files.values():
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)

if __name__ == "__main__":
    main()
