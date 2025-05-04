import json
import base64
from io import BytesIO
from PIL import Image
# MongoDB connection 
# Path to the file
file_path = 'resp.txt'
def decode_base64_uri_to_pil(base64_data):
    image_data = base64.b64decode(base64_data.split(",")[1])  # Remove 'data:image/...;base64,' part
    return Image.open(BytesIO(image_data))
# Read and process the file
with open(file_path, 'r') as file:
    for line in file:
        # Parse each line as JSON
        data = json.loads(line)
        img = decode_base64_uri_to_pil(data['img'])
        img.save("img.jpg")
        print(img)
        #     print(collection.find_one({"user_id": "110150947229138112187"}))
        #     # Insert the 'img' field into MongoDB if it exists
        #     if 'img' in data:
        #         result = collection.update_one(
        #             {'_id': ObjectId('6816612f52cd6e8b120b6526')},  # Replace with your document ID
        #             {'$set': {'image': data['img']}}
        #         )
        #         if result.matched_count > 0:
        #             print(f"Updated document with _id: 6816612f52cd6e8b120b6526")
        #         else:
        #             print(f"No document found with _id: 6816612f52cd6e8b120b6526")
        # except json.JSONDecodeError:
        #     print("Invalid JSON:", line)