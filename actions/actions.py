# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import pipeline
import os, re
from dotenv import load_dotenv

# -------------------------
# Setup
# -------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY environment variable.")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "medicalbot"
index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# HuggingFace pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# -------------------------
# Helpers
# -------------------------
def clean_text(text: str) -> str:
    """Remove junk artifacts from Pinecone text."""
    text = re.sub(r"CMDT\d+.*", "", text)
    text = re.sub(r"\b\d{1,3}[–-]?\d*\b", "", text)
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_info(raw_text: str):
    """Summarize + classify into What it is, Causes, Symptoms, Prevention."""
    cleaned = clean_text(raw_text)

    # Summarize for "What it is"
    try:
        input_len = len(cleaned.split())
        max_len = min(input_len - 5, 100)
        min_len = min(30, max_len // 2)
        summary = summarizer(cleaned, max_length=max_len, min_length=min_len, do_sample=False)
        what_it_is = summary[0]["summary_text"]
    except Exception:
        what_it_is = cleaned[:300]

    sentences = re.split(r'(?<=[.!?]) +', cleaned)
    categories = {"Causes": [], "Symptoms": [], "Prevention/Treatment": []}
    candidate_labels = ["cause", "symptom", "treatment", "prevention"]

    for sent in sentences:
        if len(sent.split()) < 4:
            continue
        try:
            result = classifier(sent, candidate_labels)
            best_label = result["labels"][0]
            if best_label == "cause":
                categories["Causes"].append(sent)
            elif best_label == "symptom":
                categories["Symptoms"].append(sent)
            elif best_label in ["treatment", "prevention"]:
                categories["Prevention/Treatment"].append(sent)
        except Exception:
            continue

    return {
        "What it is": what_it_is,
        "Causes": " ".join(categories["Causes"]) or "Not clearly specified in the source.",
        "Symptoms": " ".join(categories["Symptoms"]) or "Not clearly specified in the source.",
        "Prevention/Treatment": " ".join(categories["Prevention/Treatment"]) or "Not clearly specified in the source."
    }

def detect_intent_fallback(query: str) -> str:
    """Fallback intent classification with HuggingFace."""
    candidate_intents = ["ask_disease_info", "ask_symptoms", "ask_prevention", "general"]
    result = classifier(query, candidate_intents)
    return result["labels"][0]

# -------------------------
# Twilio-safe send
# -------------------------
def send_safe(dispatcher, text: str, max_len: int = 900):
    """Split and sanitize text for Twilio WhatsApp replies."""
    text = re.sub(r"[*_`]+", "", text)  # remove markdown
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]
    for c in chunks:
        dispatcher.utter_message(text=c.strip())

# -------------------------
# Actions
# -------------------------
class ActionRetrieveMedicalInfo(Action):
    def name(self) -> Text:
        return "action_retrieve_medical_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.latest_message.get("text")
        rasa_intent = tracker.latest_message["intent"]["name"]

        if not query:
            send_safe(dispatcher, "Sorry, I didn’t catch that. Could you repeat?")
            return []

        intent = detect_intent_fallback(query) or rasa_intent

        # Embed query
        q_embed = embed_model.encode(query).tolist()

        # Query Pinecone
        try:
            results = index.query(vector=q_embed, top_k=1, include_metadata=True)
        except Exception as e:
            send_safe(dispatcher, f"Error querying knowledge base: {str(e)}")
            return []

        if not results or not results.matches:
            send_safe(dispatcher, "Sorry, I couldn’t find anything relevant in my medical knowledge base.")
            return []

        raw_text = results.matches[0].metadata.get("text", "")
        info = extract_info(raw_text)

        # Build response
        if intent == "ask_disease_info":
            answer = f"Here’s what I found:\n\nWhat it is: {info['What it is']}"
        elif intent == "ask_symptoms":
            answer = f"Here’s what I found:\n\nSymptoms: {info['Symptoms']}"
        elif intent == "ask_prevention":
            answer = f"Here’s what I found:\n\nPrevention/Treatment: {info['Prevention/Treatment']}"
        else:
            answer = (
                f"Here’s what I found:\n\n"
                f"What it is: {info['What it is']}\n\n"
                f"Causes: {info['Causes']}\n\n"
                f"Symptoms: {info['Symptoms']}\n\n"
                f"Prevention/Treatment: {info['Prevention/Treatment']}"
            )

        answer += "\n\nDisclaimer: This is educational information, not a substitute for professional medical advice."
        send_safe(dispatcher, answer)
        return []

# -------------------------------
# Symptom checker actions
# -------------------------------
symptom_classifier = pipeline("text-classification", model="Zabihin/Symptom_to_Diagnosis")
explanation_generator = pipeline("text2text-generation", model="google/flan-t5-large")

class ActionSymptomChecker(Action):
    def name(self) -> Text: return "action_symptom_checker"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        prediction = symptom_classifier(user_input, truncation=True)
        disease = prediction[0]["label"]
        confidence = prediction[0]["score"]

        msg = f"Based on your symptoms:\nDisease: {disease}\nConfidence: {confidence:.2f}\n\nDisclaimer: Educational info only."
        send_safe(dispatcher, msg.strip())
        return []

class ActionSymptomCheckerExtractive(Action):
    def name(self) -> Text: return "action_symptom_checker_extractive"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        prediction = symptom_classifier(user_input, truncation=True)
        disease = prediction[0]["label"]
        confidence = prediction[0]["score"]

        prompt = f"""
        Patient reports: {user_input}.
        Predicted disease: {disease}.
        Write a structured medical note.
        Format:
        Disease:
        Possible Causes:
        Suggested Treatments:
        Common Medicines:
        Prevention Tips:
        """

        try:
            response = explanation_generator(prompt, max_new_tokens=256, do_sample=False)
            advice = response[0]["generated_text"]
        except Exception:
            advice = f"Disease: {disease}\nPossible Causes: Infection\nSuggested Treatments: Rest\nCommon Medicines: Paracetamol\nPrevention Tips: Healthy diet"

        msg = f"Based on your symptoms:\nDisease: {disease}\nConfidence: {confidence:.2f}\n\n{advice}\n\nDisclaimer: Educational info only."
        send_safe(dispatcher, msg.strip())
        return []








'''
class ActionVaccinationSchedule(Action):
    def name(self): return "action_vaccination_schedule"

    def run(self, dispatcher, tracker, domain):
        user_dob = tracker.get_slot("dob")
        if not user_dob:
            dispatcher.utter_message("Please provide your child's date of birth.")
            return []
        
        # Example: query your vaccination DB
        next_vaccine = get_next_vaccine(user_dob)  # custom function
        
        dispatcher.utter_message(f"📅 Next vaccine due: {next_vaccine['name']} on {next_vaccine['date']}.")
        return []


class ActionFacilityLookup(Action):
    def name(self): 
        return "action_facility_lookup"

    def run(self, dispatcher, tracker, domain):
        user_location = tracker.get_slot("location")
        if not user_location:
            dispatcher.utter_message("Please provide your pincode or location.")
            return []

        nearest = find_nearest_facility(user_location)
        dispatcher.utter_message(
            f"🏥 Nearest facility: {nearest['name']}, {nearest['address']}"
        )
        return []




class ActionOutbreakAlert(Action):
    def name(self): return "action_outbreak_alert"

    def run(self, dispatcher, tracker, domain):
        user_district = tracker.get_slot("district")
        outbreaks = get_recent_outbreaks(user_district)  # query your outbreak_event table
        if outbreaks:
            msg = "⚠️ Current outbreaks:\n" + "\n".join([f"- {o['disease']} ({o['date']})" for o in outbreaks])
        else:
            msg = "✅ No active outbreaks in your district."
        dispatcher.utter_message(msg)
        return []


class ActionEmergencyHandoff(Action):
    def name(self): return "action_emergency_handoff"
    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("🚑 This seems urgent. Please call 108 (emergency ambulance).")
        return []
'''