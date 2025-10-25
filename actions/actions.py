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


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher

from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import pipeline
import os, re, wikipedia
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
    if not cleaned:
        return None

    # Summarize for "What it is"
    try:
        input_len = len(cleaned.split())
        max_len = min(max(input_len - 5, 30), 100)
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
        "What it is": what_it_is or "Not clearly specified in the source.",
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

        # Embed query and try Pinecone
        q_embed = embed_model.encode(query).tolist()
        try:
            results = index.query(vector=q_embed, top_k=1, include_metadata=True)
        except Exception as e:
            results = None

        raw_text = None
        if results and results.matches:
            raw_text = results.matches[0].metadata.get("text", "")

        # Fallback to Wikipedia if nothing useful
        info = extract_info(raw_text) if raw_text else None
        if not info or info["What it is"].startswith("Not clearly specified"):
            try:
                wiki_summary = wikipedia.summary(query, sentences=5, auto_suggest=True, redirect=True)
                info = extract_info(wiki_summary)
            except Exception:
                send_safe(dispatcher, "Sorry, I couldn’t find reliable information in my sources or Wikipedia.")
                return []

        # Build response
        if intent in ["ask_disease_info", "general"]:
            answer = (
                f"Here’s what I found:\n\n"
                f"📝 What it is: {info['What it is']}\n\n"
                f"🧪 Causes: {info['Causes']}\n\n"
                f"🤒 Symptoms: {info['Symptoms']}\n\n"
                f"💊 Prevention/Treatment: {info['Prevention/Treatment']}"
            )
        elif intent == "ask_symptoms":
            answer = f"Here’s what I found:\n\n🤒 Symptoms: {info['Symptoms']}"
        elif intent == "ask_prevention":
            answer = f"Here’s what I found:\n\n💊 Prevention/Treatment: {info['Prevention/Treatment']}"
        else:
            answer = f"Here’s what I found:\n\n📝 What it is: {info['What it is']}"

        answer += "\n\n Disclaimer: This is educational info only, not a substitute for professional medical advice."
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

        conf_percent = confidence * 100
        msg = f"Based on your symptoms:\nPredicted Disease: {disease}\nConfidence: {conf_percent:.0f}%\n\n Disclaimer: This isn't a medical diagnosis, only general guidance. Please visit a doctor if things get serious. Can you share a few more symptoms you're experiencing so I can respond more accurately."
        send_safe(dispatcher, msg.strip())

        # ✅ Specific recommendations (if available)
        specific_recommendations = {
            "flu": "Stay hydrated, rest well, and monitor your fever. If symptoms worsen, visit a doctor.",
            "migraine": "Rest in a dark, quiet room. Avoid triggers like bright light or loud sounds.",
            "COVID-19": "Isolate yourself, wear a mask, monitor oxygen levels, and seek testing.",
            "common cold": "Drink warm fluids, rest, and use saline nasal sprays. See a doctor if symptoms persist.",
            "malaria": "Seek prompt testing & prescription antimalarials from a clinician. Stay hydrated and rest.",
            "dengue": "Rest, hydrate well, use paracetamol for fever. Avoid aspirin/NSAIDs. Monitor warning signs.",
            "heart attack": "Call emergency services immediately. Do not drive yourself. Follow your doctor's guidance.",
            "diabetes": "Monitor blood sugar regularly, maintain a balanced diet, and follow your doctor's recommendations.",
            "hypertension": "Reduce salt intake, manage stress, exercise moderately, and follow prescribed medication.",
            "asthma": "Use inhalers as prescribed, avoid triggers, and monitor breathing patterns.",
            "food poisoning": "Stay hydrated with oral rehydration solutions, rest, and seek medical care if severe.",
            "gastroenteritis": "Drink plenty of fluids, rest, and follow a bland diet. Seek help if dehydration occurs.",
            "chickenpox": "Rest, stay hydrated, and avoid scratching rashes. Consult a doctor for severe cases.",
            "measles": "Rest, stay hydrated, and manage fever with paracetamol. Seek medical care if complications arise.",
            "tonsillitis": "Rest, hydrate, gargle warm salt water, and follow medical advice if severe.",
            "urinary Tract Infection": "Drink plenty of water, rest, and seek medical attention for antibiotics if needed.",
            "bronchitis": "Rest, stay hydrated, and avoid irritants like smoke. Consult a doctor if symptoms worsen.",
            "anemia": "Eat iron-rich foods, stay hydrated, and follow your doctor's guidance for supplements.",
            "hypothyroidism": "Take prescribed medication, maintain a balanced diet, and follow up regularly with your doctor.",
            "hyperthyroidism": "Take prescribed medication, avoid excess iodine, and monitor symptoms with your healthcare provider.",
            "allergies": "Avoid known triggers, take prescribed antihistamines, and seek medical advice for severe reactions.",
            "arthritis": "Stay active with gentle exercises, maintain a healthy weight, apply heat or cold packs to affected joints, and follow your doctor's guidance on medications or therapy."
        }
        if disease in specific_recommendations:
            advice = specific_recommendations[disease]
            dispatcher.utter_message(text=f"💡 Recommendation: {advice}")

        # ✅ General recommendations (always shown)
        general_advice = (
            "General Health Tips:\n"
            "• Drink plenty of water\n"
            "• Take adequate rest\n"
            "• Eat light, nutritious meals\n"
            "• Avoid self-medication\n"
            "• Consult a doctor if symptoms get worse"
        )
        dispatcher.utter_message(text=general_advice)

        # ✅ One-Tap Emergency Support
        dispatcher.utter_message(
            text=(
                "Need urgent help?\n\n"
                "Reply with Help"
                # "1 for  Call Ambulance\n"
                # "2 for  Find Nearest Hospital"
            )
        )
        return []

class ActionSymptomCheckerExtractive(Action):
    def name(self) -> Text: return "action_symptom_checker_extractive"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get("text")
        prediction = symptom_classifier(user_input, truncation=True)
        disease = prediction[0]["label"]
        confidence = prediction[0]["score"]

        conf_percent = confidence * 100

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

        msg = f"Based on your symptoms:\nDisease: {disease}\nConfidence: {conf_percent:.0f}%\n\n{advice}\n\n⚠️Disclaimer: Educational info only."
        send_safe(dispatcher, msg.strip())
        return []
    

import googlemaps
from twilio.rest import Client

# API Keys
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")  # Your Twilio sandbox or business number

gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

class ActionEmergencyHelp(Action):
    def name(self):
        return "action_emergency_help"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_location = tracker.get_slot("location")
        user_phone = tracker.sender_id   # Twilio WhatsApp automatically sets this

        if not user_location:
            dispatcher.utter_message("🏥 Please share your location (city or pincode + country).")
            return []

        # Step 1: Geocode location
        try:
            geocode = gmaps.geocode(user_location)
            if not geocode:
                dispatcher.utter_message("❌ Location not found. Try again with city + country.")
                return []
            lat, lng = geocode[0]["geometry"]["location"].values()
        except Exception as e:
            dispatcher.utter_message("❌ Error with location lookup.")
            print("Google Maps Error:", e)
            return []

        # Step 2: Find nearby hospital & ambulance
        hospital = "Nearest Hospital"
        ambulance = "108 Emergency Ambulance"
        try:
            hospitals = gmaps.places_nearby(location=(lat, lng), radius=3000, type="hospital")
            if hospitals.get("results"):
                hospital = hospitals["results"][0]["name"]

            ambulances = gmaps.places_nearby(location=(lat, lng), radius=5000, keyword="ambulance")
            if ambulances.get("results"):
                ambulance = ambulances["results"][0]["name"]
        except Exception as e:
            print("Places API Error:", e)

        # Step 3: Send WhatsApp via Twilio
        message_body = f"🚨 Emergency Alert!\n🚑 Ambulance: {ambulance}\n🏥 Hospital: {hospital}\n📍 Location: {user_location}"
        try:
            twilio_client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE,
                to=user_phone
            )
        except Exception as e:
            dispatcher.utter_message(f"⚠️ Failed to send WhatsApp: {e}")

        # Step 4: Confirm in chat
        dispatcher.utter_message(
            text=f"✅ Help request processed!\n🚑 Ambulance: {ambulance}\n🏥 Hospital notified: {hospital}\n📞 Confirmation sent to you via WhatsApp."
        )
        return []

#

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import Restarted, EventType, SlotSet

from main import  Dose_Availability_Pincode

class ValidatepincodeForm(FormValidationAction):
    def name(self) -> Text:
        return "slot_pincode_form"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        required_slots = ["pincode", "date"]
        # "job_after_exit", "job_type", "acquire_skill", "skill_type", "any_business", "business_venture", "need_loan"

        for slot_name in required_slots:
            if tracker.slots.get(slot_name) is None:
                # The slot is not filled yet. Request the user to fill this slot next.
                return [SlotSet("requested_slot", slot_name)]

        return [SlotSet("requested_slot", None)]


class ActionPincodeSubmit(Action):

    def name(self) -> Text:
        return "action_pincode_submit"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global message
        message=Dose_Availability_Pincode(tracker.get_slot('pincode'),tracker.get_slot('date'))
        dispatcher.utter_message(text=message)
        buttons = [
            {'payload': "/affirm", 'title': "Yes"},
            {'payload': "/deny", 'title': "No"},
        ]
        dispatcher.utter_message(text="Would you like to get the details on your email id?",buttons=buttons)

        return []


# class ActionSendEmail(Action):

#     def name(self) -> Text:
#         return "action_send_mail"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         send_email(tracker.get_slot("email"),message)
#         dispatcher.utter_message(text="We have successfully sent the mail to your Email ID: {}".format(tracker.get_slot("email")))

#         return []

# class ActionRestart(Action):

#     def name(self) -> Text:
#       return "action_restart"

#     async def run(
#       self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
#     ) -> List[Dict[Text, Any]]:

#       return [Restarted()]


'''
import googlemaps
from rasa_sdk.events import SlotSet
from twilio.rest import Client

# API Keys from environment
GOOGLE_MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")  # e.g. "whatsapp:+14155238886"

# Init clients
gmaps = googlemaps.Client(key=GOOGLE_MAPS_KEY)
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)


class ActionEmergencyHelp(Action):
    def name(self):
        return "action_emergency_help"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_location = tracker.get_slot("location")
        user_phone = tracker.get_slot("phone")

        if not user_location:
            dispatcher.utter_message("🏥 Please share your location (city or pincode).")
            return []
        if not user_phone:
            dispatcher.utter_message("📞 Please provide your phone number so we can notify you.")
            return []

        # ✅ Step 1: Geocode location
        try:
            query_location = user_location.strip()
            if "india" not in query_location.lower():
                query_location = f"{query_location}, India"

            geocode = gmaps.geocode(query_location)
            if not geocode:
                dispatcher.utter_message("❌ Location not found. Try again with city or pincode + country.")
                return []
            lat, lng = geocode[0]["geometry"]["location"].values()
        except Exception as e:
            dispatcher.utter_message("❌ Error with location lookup.")
            print("Google Maps Error:", e)
            return []

        # ✅ Step 2: Find nearby hospital + ambulance
        try:
            hospital = gmaps.places_nearby(location=(lat, lng), radius=3000, type="hospital")
            ambulance = gmaps.places_nearby(location=(lat, lng), radius=5000, keyword="ambulance")

            hospital_name = hospital["results"][0]["name"] if hospital.get("results") else "Nearest Hospital"
            ambulance_name = ambulance["results"][0]["name"] if ambulance.get("results") else "108 Emergency Ambulance"
        except Exception as e:
            dispatcher.utter_message("❌ Error while finding nearby services.")
            print("Places API Error:", e)
            hospital_name = "Nearest Hospital"
            ambulance_name = "108 Emergency Ambulance"

        # ✅ Step 3: WhatsApp Message via Twilio
        message_body = (
            f"🚨 *Emergency Alert!*\n\n"
            f"🚑 Ambulance booked: {ambulance_name}\n"
            f"🏥 Hospital notified: {hospital_name}\n"
            f"📍 Location: {user_location}\n\n"
            f"Please stay calm. Help is on the way. Dial *108* for direct ambulance hotline in India."
        )

        try:
            twilio_client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE,
                to=f"whatsapp:{user_phone}" if not user_phone.startswith("whatsapp:") else user_phone
            )
        except Exception as e:
            dispatcher.utter_message(f"⚠️ Failed to send WhatsApp message: {e}")

        # ✅ Step 4: Confirm in Chat
        dispatcher.utter_message(
            f"✅ Help request processed!\n"
            f"🚑 Ambulance: {ambulance_name}\n"
            f"🏥 Hospital notified: {hospital_name}\n"
            f"📞 Confirmation sent to {user_phone} via WhatsApp."
        )

        return [SlotSet("hospital", hospital_name), SlotSet("ambulance", ambulance_name)]
'''




'''
class ActionSetRequestType(Action):
    def name(self) -> Text:
        return "action_set_request_type"

    def run(self, dispatcher, tracker, domain):
        intent = tracker.latest_message['intent'].get('name')

        if intent == "emergency_hospital":
            return [
                SlotSet("request_type", "hospital"),
                SlotSet("location", None)   # reset old location
            ]
        elif intent == "emergency_ambulance":
            return [
                SlotSet("request_type", "ambulance"),
                SlotSet("location", None)   # reset old location
            ]
        return []

class ActionFindService(Action):
    """Unified action for hospital and ambulance based on request_type slot."""

    def name(self) -> Text:
        return "action_find_service"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        request_type = tracker.get_slot("request_type")
        user_location = tracker.get_slot("location")

        if not user_location:
            msg = "🏥 Please share your location (city or pincode) to find the nearest hospital." if request_type == "hospital" else "🚑 Please share your location (city or pincode) to find ambulance services."
            dispatcher.utter_message(text=msg)
            return []

        if not request_type:
            dispatcher.utter_message(text="❌ Something went wrong. Please specify if you need a hospital or ambulance.")
            return []

        try:
            # Step 1: Geocode location
            geocode = gmaps.geocode(user_location)
            if not geocode:
                dispatcher.utter_message(
                    text=f"❌ Sorry, I couldn’t find {user_location}. Please try another location."
                )
                return []

            lat, lng = geocode[0]["geometry"]["location"].values()

            # Step 2: Search based on request_type
            if request_type == "hospital":
                places = gmaps.places_nearby(
                    location=(lat, lng),
                    radius=3000,
                    type="hospital"
                )
                if not places.get("results"):
                    dispatcher.utter_message(text=f"⚠️ No hospitals found near {user_location}.")
                    return []

                hospitals = places["results"][:3]
                reply = f"🏥 Nearest hospitals near *{user_location}*:\n"
                for h in hospitals:
                    name = h["name"]
                    address = h.get("vicinity", "Address not available")
                    rating = h.get("rating", "N/A")
                    reply += f"\n• {name} (⭐ {rating})\n📍 {address}\n"

            elif request_type == "ambulance":
                places = gmaps.places_nearby(
                    location=(lat, lng),
                    radius=5000,
                    keyword="ambulance"
                )
                results = places.get("results", [])
                if not results:
                    reply = (
                        f"🚑 No dedicated ambulance providers found near *{user_location}*.\n"
                        f"👉 Please dial **108** immediately for emergency ambulance service (India)."
                    )
                else:
                    top_services = results[:3]
                    reply = f"🚑 Ambulance services near *{user_location}*:\n"
                    for amb in top_services:
                        name = amb["name"]
                        address = amb.get("vicinity", "Address not available")
                        rating = amb.get("rating", "N/A")
                        reply += f"\n• {name} (⭐ {rating})\n📍 {address}\n"
                    reply += "\n⚠️ For emergencies, always dial **108** first."

            dispatcher.utter_message(text=reply.strip())

        except Exception as e:
            dispatcher.utter_message(
                text="❌ Sorry, something went wrong while fetching data. Please try again later."
            )
            print("Error in ActionFindService:", e)

        return []
'''

