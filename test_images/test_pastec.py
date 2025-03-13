#\!/usr/bin/env python3
import requests
import sys
import os
import json
import tempfile
import random
from PIL import Image, ImageDraw

def create_test_image(filename, shape="rectangle"):
    """Create a test image with a specific shape"""
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw different shapes
    if shape == "rectangle":
        draw.rectangle([(100, 100), (700, 500)], fill=(255, 0, 0))
    elif shape == "circle":
        draw.ellipse([(150, 100), (650, 500)], fill=(0, 255, 0))
    elif shape == "triangle":
        draw.polygon([(400, 100), (100, 500), (700, 500)], fill=(0, 0, 255))
    
    # Add some small features for ORB to detect
    for i in range(10):
        x = random.randint(200, 600)
        y = random.randint(200, 400)
        draw.rectangle([(x-10, y-10), (x+10, y+10)], fill=(0, 0, 0))
    
    img.save(filename)
    return filename

def test_pastec(host="localhost", port=4213):
    """Test the Pastec API by adding and searching for images"""
    base_url = f"http://{host}:{port}"
    
    print("Testing Pastec API...")
    
    # Create temp directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        image_files = [
            create_test_image(os.path.join(temp_dir, "rectangle.jpg"), "rectangle"),
            create_test_image(os.path.join(temp_dir, "circle.jpg"), "circle"),
            create_test_image(os.path.join(temp_dir, "triangle.jpg"), "triangle")
        ]
        
        image_ids = []
        
        # 1. Clear any existing index
        print("\n--- Clearing index ---")
        clear_response = requests.post(f"{base_url}/index/io", json={"type": "CLEAR"})
        print(f"Clear index response: {clear_response.json()}")
        
        # 2. Add images to index
        print("\n--- Adding images to index ---")
        for i, image_file in enumerate(image_files):
            image_id = i + 1
            image_ids.append(image_id)
            
            with open(image_file, 'rb') as f:
                image_data = f.read()
            
            add_response = requests.post(f"{base_url}/index/images/{image_id}", 
                                         data=image_data)
            print(f"Added image {image_id} ({os.path.basename(image_file)}): {add_response.json()}")
            
            # Add tags to images
            tag = os.path.basename(image_file).split('.')[0]  # Use filename as tag
            tag_response = requests.post(f"{base_url}/index/images/{image_id}/tag", 
                                         data=tag)
            print(f"Added tag '{tag}' to image {image_id}: {tag_response.json()}")
        
        # 3. List all image IDs
        print("\n--- Listing all image IDs ---")
        list_response = requests.get(f"{base_url}/index/imageIds")
        print(f"List images response: {list_response.json()}")
        
        # 4. Search for each image
        print("\n--- Searching for images ---")
        for image_file in image_files:
            with open(image_file, 'rb') as f:
                image_data = f.read()
            
            search_response = requests.post(f"{base_url}/index/searcher", 
                                           data=image_data)
            print(f"Search results for {os.path.basename(image_file)}: {json.dumps(search_response.json(), indent=2)}")
            
        # 5. Create a slightly modified version of an image and search for it
        print("\n--- Searching with a modified image ---")
        modified_file = os.path.join(temp_dir, "modified_rectangle.jpg")
        img = Image.open(image_files[0])
        draw = ImageDraw.Draw(img)
        # Add some extra features but keep the main shape
        draw.rectangle([(300, 250), (400, 350)], fill=(0, 0, 0))
        img.save(modified_file)
        
        with open(modified_file, 'rb') as f:
            image_data = f.read()
        
        search_response = requests.post(f"{base_url}/index/searcher", 
                                       data=image_data)
        print(f"Search results for modified image: {json.dumps(search_response.json(), indent=2)}")
            
        # 6. Delete one image
        print("\n--- Deleting an image ---")
        delete_id = image_ids[0]
        delete_response = requests.delete(f"{base_url}/index/images/{delete_id}")
        print(f"Delete image {delete_id} response: {delete_response.json()}")
        
        # 7. Verify deletion by listing images
        list_response = requests.get(f"{base_url}/index/imageIds")
        print(f"Updated list of images: {list_response.json()}")
        
        print("\nTest completed\!")

if __name__ == "__main__":
    host = "localhost"
    port = 4213  # Use the port we set in docker-compose.yml
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
        
    test_pastec(host, port)
