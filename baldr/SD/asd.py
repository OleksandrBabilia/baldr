import json
from pymongo import MongoClient
from bson.objectid import ObjectId
# MongoDB connection details
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['chatdb']
collection = db['messages']  # Replace with your collection name
doc = collection.find_one()
print(doc)
# Path to the file
file_path = 'resp.txt'

# Read and process the file
with open(file_path, 'r') as file:
    for line in file:
        try:
            # Parse each line as JSON
            data = json.loads(line)
            print(collection.find_one({"user_id": "110150947229138112187"}))
            # Insert the 'img' field into MongoDB if it exists
            if 'img' in data:
                result = collection.update_one(
                    {'_id': ObjectId('6816612f52cd6e8b120b6526')},  # Replace with your document ID
                    {'$set': {'image': data['img']}}
                )
                if result.matched_count > 0:
                    print(f"Updated document with _id: 6816612f52cd6e8b120b6526")
                else:
                    print(f"No document found with _id: 6816612f52cd6e8b120b6526")
        except json.JSONDecodeError:
            print("Invalid JSON:", line)