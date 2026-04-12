import json
import os
from datetime import datetime, timedelta
import random
from langchain_core.documents import Document

def generate_mock_drone_descriptions(json_path: str, num_records: int = 50):
    """Generates mock drone visual descriptions simulating VLM output."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    start_time = datetime(2013, 9, 12, 8, 0)
    records = []
    
    locations = ["Highway 36", "Downtown Boulder", "Jamestown", "Lyons Bridge"]
    descriptions = [
        "Aerial view shows widespread flooding. The main bridge is completely submerged under muddy water. Debris is piled up against the supports.",
        "Drone footage reveals a collapsed residential structure with severe water damage. Emergency vehicles are visible in the distance but cannot access the area due to flooded roads.",
        "Image shows a swollen river overflowing its banks. A vehicle is stranded in the water. No individuals are visible inside or around the vehicle.",
        "Overhead view of an evacuation route. The road surface has severely eroded, creating a massive sinkhole. The route is impassable for standard vehicles."
    ]
    
    for i in range(num_records):
        timestamp = start_time + timedelta(minutes=30 * i)
        loc = random.choice(locations)
        desc = random.choice(descriptions)
        
        records.append({
            "image_id": f"IMG_{1000+i}",
            "timestamp": timestamp.isoformat(),
            "location": loc,
            "vlm_description": f"At {loc}, {desc}"
        })
        
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=4)
    print(f"Generated {num_records} mock drone descriptions at {json_path}")

def load_drone_data(json_path: str) -> list[Document]:
    """Loads JSON drone descriptions into Langchain Documents."""
    if not os.path.exists(json_path):
        print(f"Drone data not found at {json_path}. Generating mock data...")
        generate_mock_drone_descriptions(json_path)
        
    with open(json_path, 'r') as f:
        records = json.load(f)
        
    documents = []
    for row in records:
        description = row.get('observations', row.get('vlm_description', 'No description'))
        image_id = row.get('drone_id', row.get('image_id', 'Unknown'))
        
        text = f"DRONE OBSERVATION mapped at {row['timestamp']}: {description}"
        metadata = {
            "source": "Drone_VLM",
            "image_id": image_id,
            "location": row['location'],
            "modality": "imagery_description",
            "timestamp": row['timestamp']
        }
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
            
    print(f"Loaded {len(documents)} real drone observation documents.")
    return documents

if __name__ == "__main__":
    docs = load_drone_data(os.path.join("data", "imagery", "drone_metadata.json"))
    for d in docs[:3]:
        print(d.page_content)
