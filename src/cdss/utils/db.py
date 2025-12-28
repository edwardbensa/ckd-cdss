"""MongoDB utility functions."""
from datetime import date, datetime, time
from pymongo import MongoClient
import streamlit as st
from src.cdss.config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME

@st.cache_resource
def get_mongodb_connection():
    '''Connect to MongoDB and return database'''
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    return db

def get_patients_collection():
    '''Get patients collection'''
    db = get_mongodb_connection()
    return db[COLLECTION_NAME]

def clean(doc):
    """Remove null values and normalise dates."""
    for k, v in doc.items():
        if isinstance(v, date) and not isinstance(v, datetime):
            doc[k] = datetime.combine(v, time.min)
    clean_data = {k: v for k, v in doc.items() if v not in (None, "", [])}
    return clean_data

def add_patient(patient_data):
    '''Add a new patient to the database'''
    collection = get_patients_collection()
    clean_data = clean(patient_data)
    result = collection.insert_one(clean_data)
    return result.inserted_id

def get_patient(patient_id):
    '''Get a single patient by ID'''
    collection = get_patients_collection()
    return collection.find_one({"patient_id": patient_id})

def get_all_patients(query=None):
    '''Get all patients matching query'''
    collection = get_patients_collection()
    if query is None:
        query = {}
    return list(collection.find(query))

def update_patient(patient_id, update_data):
    '''Update patient information'''
    collection = get_patients_collection()
    clean_data = {k: v for k, v in update_data.items() if v not in (None, "", [])}
    result = collection.update_one(
        {"patient_id": patient_id},
        {"$set": clean_data}
    )
    return result.modified_count

def delete_patient(patient_id):
    '''Delete a patient'''
    collection = get_patients_collection()
    result = collection.delete_one({"patient_id": patient_id})
    return result.deleted_count

def patient_exists(patient_id):
    '''Check if patient exists'''
    collection = get_patients_collection()
    return collection.find_one({"patient_id": patient_id}) is not None

def next_patient_id():
    """Get the next patient id after the last."""
    patients = get_all_patients()
    if not patients:
        return 1

    ids = []
    for p in patients:
        try:
            ids.append(int(p.get("patient_id")))
        except (TypeError, ValueError):
            continue

    return max(ids) + 1 if ids else 1

def serialize_mongo(data):
    """Recursively convert ObjectId and datetime to string."""
    if isinstance(data, list):
        return [serialize_mongo(item) for item in data]
    if isinstance(data, dict):
        return {k: serialize_mongo(v) for k, v in data.items()}
    if isinstance(data, datetime):
        return data.isoformat()
    if hasattr(data, "__str__") and "ObjectId" in str(type(data)):
        return str(data)
    return data
