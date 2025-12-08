import pyrealsense2 as rs
import numpy as np
import pandas as pd
import cv2
import requests
import datetime
import json
import os
from flask import Flask, render_template_string, jsonify
import base64
from threading import Lock

# ==================== CONFIG ====================
AZURE_LLAMAVISION_API_URL = "https://Llama-3-2-90B-Vision-Instruct-mx.eastus2.models.ai.azure.com"
AZURE_LLAMAVISION_API_KEY = os.getenv("AZURE_LLAMAVISION_API_KEY")
DEFAULT_DENSITY = 700  # Default fallback density in kg/m³
TREES_ABSORPTION_CO2 = 21  # kg CO₂ per year per tree
FOOD_STACKING_FACTORS = {
    "rice": 1.1,  # Slight stacking (10% increase)
    "pasta": 1.1,  # Similar to rice
    "bread": 1.3,  # Moderate stacking (30% increase)
    "cheese": 1.2,  # Slight compression
    "pizza": 1.5,  # Rigid, can stack high (50% increase)
    "meat": 1.3,  # Moderate stacking
    "vegetables": 1.2,  # Some compression
    "fruits": 1.2,  # Irregular stacking
    "bones": 1.6,  # Hard, stacks easily
    "cereal": 1.1,  # Slight stacking, lightweight
    "eggshells": 1.3,  # Fragile but moderate stacking
    "fish": 1.4,  # Can be stacked but compresses slightly
    "nutshells": 1.5,  # Rigid stacking
    "onion-peels": 1.2,  # Light stacking
    "potatoes": 1.3,  # Heavier, moderate stacking
    "tomatoes": 1.2,  # Soft, slightly compressible
    "carrots": 1.3,  # Dense stacking
    "corncobs": 1.5,  # Rigid stacking
    "watermelon-rind": 1.2,  # Large but can compress slightly
    "grapes": 1.2,  # Slight stacking, fragile
    "lettuce": 1.1,  # Compresses easily
    "unknown": 1.0  # Default, no adjustment
}


# ==================== REALSENSE INIT ====================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ==================== FLASK SERVER ====================
app = Flask(__name__)

# Add after other global variables
waste_data_lock = Lock()
current_waste_data = {
    "total_weight": 0,
    "total_co2": 0,
    "total_methane": 0,
    "total_water": 0,
    "total_energy": 0,
    "total_land_use": 0,
    "latest_food": "No waste detected yet",
    "goal_message": "Help us reach our goal!"
}
SESSION_TIMEOUT = 30  # 30 seconds timeout
manual_reset = False

@app.route('/')
def index():
    return render_template_string('''
    <html>
    <head>
        <title>Food Waste Tracker</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .stats-container {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .stat-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 15px;
                padding: 10px;
                border-bottom: 1px solid #eee;
            }
            .stat-label {
                font-weight: bold;
                color: #34495e;
            }
            .stat-value {
                color: #2c3e50;
            }
            .goal-container {
                margin-top: 30px;
                text-align: center;
                padding: 20px;
                background-color: #e8f5e9;
                border-radius: 10px;
            }
            .goal-message {
                color: #2c3e50;
                font-size: 1.2em;
                margin: 0;
            }
        </style>
        <script>
            function updateData() {
                fetch('/latest_waste')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('food').innerText = data.latest_food;
                        document.getElementById('weight').innerText = data.latest_weight + " kg";
                        document.getElementById('CO2').innerText = data.latest_CO2 + " kg CO₂ (losing " + data.trees_lost + " trees)";
                        document.getElementById('methane').innerText = data.methane_emissions + " kg CH₄ (high greenhouse impact)";
                        document.getElementById('water').innerText = data.water_usage + " L (wasted " + data.showers_wasted + " showers)";
                        document.getElementById('energy').innerText = data.energy_usage + " MJ (laptop used for " + data.laptop_hours + " hours)";
                        document.getElementById('goal-message').innerText = data.goal_message;
                    })
                    .catch(error => console.error('Error:', error));
            }
            setInterval(updateData, 3000);
        </script>
    </head>
    <body onload="updateData()">
        <h1>Food Waste Tracker</h1>
        <div class="stats-container">
            <div class="stat-row">
                <span class="stat-label">Latest Waste:</span>
                <span class="stat-value" id="food">N/A</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Weight:</span>
                <span class="stat-value" id="weight">0 kg</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">CO₂ Impact:</span>
                <span class="stat-value" id="CO2">0 kg</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Methane Emissions:</span>
                <span class="stat-value" id="methane">0 kg</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Water Used:</span>
                <span class="stat-value" id="water">0 L</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Energy Lost:</span>
                <span class="stat-value" id="energy">0 MJ</span>
            </div>
        </div>
        <div class="goal-container">
            <p class="goal-message" id="goal-message">Help us reach our goal!</p>
        </div>
    </body>
    </html>
    ''')

# ==================== FUNCTIONS ====================

CSV_FILE_PATH = "waste_data.csv"  # Change path if needed (e.g., "/home/pi/data_logs/waste_data.csv")

def initialize_csv():
    """Creates the CSV file with headers if it does not exist."""
    if not os.path.exists(CSV_FILE_PATH):
        df = pd.DataFrame(columns=[
            "Timestamp", "Food Type", "Volume (cm³)", "Density (kg/m³)", "Weight (kg)", 
            "CO₂ Emissions (kg)", "Methane Emissions (kg)", "Water Usage (L)", 
            "Energy Usage (MJ)", "Land Use (Hectares)"
        ])
        df.to_csv(CSV_FILE_PATH, mode='w', header=True, index=False)
        print(f"✅ Created {CSV_FILE_PATH} with headers.")

# Ensure CSV file is initialized at script start
initialize_csv()


def classify_food(image_base64):
    """Classify food and get environmental data using Azure Llama Vision.
    Returns a list of tuples (food_type, density, co2_emission, methane_emission, water_usage, energy_usage)."""
    if not AZURE_LLAMAVISION_API_KEY:
        print("⚠️ Azure API key not found in environment variables")
        return [("unknown", DEFAULT_DENSITY, 0, 0, 0, 0)]

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_LLAMAVISION_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "image": image_base64,
        "prompt": """Identify all food items in this image and provide structured JSON output with an array of items, where each item has:
        - classification: The name of the food waste using all lower case and dashes if space is needed
        - density: The density in kg/m³.
        - co2_emission: The CO₂ emission factor in kg CO₂ per kg of food.
        - methane_emission: The methane emission factor in kg CH₄ per kg of food.
        - water_usage: The water footprint in L per kg of food.
        - energy_usage: The energy consumption in MJ per kg of food.
        
        Return ONLY JSON output with an array of items, no extra text or explanations."""
    }

    try:
        response = requests.post(AZURE_LLAMAVISION_API_URL, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        
        try:
            result = response.json()
            # Handle both single item and array responses
            items = result if isinstance(result, list) else [result]
            
            # Process each item
            processed_items = []
            for item in items:
                # Extract values with fallbacks
                food_type = item.get("classification", "unknown").lower()
                density = float(item.get("density", DEFAULT_DENSITY))
                co2_emission = float(item.get("co2_emission", 0))
                methane_emission = float(item.get("methane_emission", 0))
                water_usage = float(item.get("water_usage", 0))
                energy_usage = float(item.get("energy_usage", 0))

                # Validate the values
                if not all(isinstance(x, (int, float)) for x in [density, co2_emission, methane_emission, water_usage, energy_usage]):
                    print(f"⚠️ Invalid numeric values in API response for {food_type}")
                    continue

                processed_items.append((food_type, density, co2_emission, methane_emission, water_usage, energy_usage))
            
            if not processed_items:
                print("⚠️ No valid items found in API response")
                return [("unknown", DEFAULT_DENSITY, 0, 0, 0, 0)]
                
            return processed_items

        except json.JSONDecodeError:
            print("⚠️ API response could not be parsed as JSON")
            return [("unknown", DEFAULT_DENSITY, 0, 0, 0, 0)]

    except requests.exceptions.Timeout:
        print("⚠️ API request timed out")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API request failed: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error in classify_food: {e}")
    
    return [("unknown", DEFAULT_DENSITY, 0, 0, 0, 0)]  # Fallback values

def get_food_volume(depth_frame, bbox):
    """Calculate the volume of food waste using depth information."""
    x, y, w, h = bbox
    
    # Get depth values within the bounding box
    depth_values = []
    
    # Sample points within the bounding box (every 5th pixel to avoid too much computation)
    for i in range(y, y + h, 5):
        for j in range(x, x + w, 5):
            if 0 <= i < depth_frame.shape[0] and 0 <= j < depth_frame.shape[1]:
                depth = depth_frame[i, j]
                if depth > 0:  # Only use valid depth values
                    depth_values.append(depth)
    
    if not depth_values:
        print("⚠️ No valid depth values found in bounding box")
        return 0
    
    # Convert to numpy array for efficient processing
    depth_values = np.array(depth_values)
    
    # Remove outliers (values beyond ±30% of the median)
    median_depth = np.median(depth_values)
    filtered_depths = depth_values[
        (depth_values > median_depth * 0.7) & 
        (depth_values < median_depth * 1.3)
    ]
    
    if len(filtered_depths) == 0:
        print("⚠️ All depth values were outliers")
        return 0
    
    # Calculate average depth in meters
    avg_depth = np.mean(filtered_depths) / 1000.0
    
    # Get the depth scale from the camera
    depth_scale = 0.001  # 1mm = 0.001m
    
    # Calculate the real-world dimensions
    real_width = w * depth_scale
    real_height = h * depth_scale
    
    # Calculate volume (assuming a rectangular prism shape)
    volume = real_width * real_height * avg_depth
    
    return volume

def estimate_weight(depth_frame, bbox, food_type):
    """Estimate the weight of the detected food item using volume and density."""
    # Calculate volume
    volume = get_food_volume(depth_frame, bbox)
    
    # Get density for this food type
    density = get_food_density_usda(food_type)
    
    # Get stacking factor for this food type
    stacking_factor = FOOD_STACKING_FACTORS.get(food_type.lower(), 1.0)
    
    # Apply stacking factor to volume (since stacking affects the effective volume)
    adjusted_volume = volume * stacking_factor
    
    # Calculate weight (mass = volume * density)
    weight = adjusted_volume * density
    
    return round(weight, 3)

def methane_decay(methane_initial, years=10, decay_rate=0.1):
    """Calculate methane emissions over time based on landfill decay rates."""
    methane_remaining = methane_initial * np.exp(-decay_rate * years)
    return round(methane_remaining, 3)

def save_to_csv(timestamp, food_type, volume, density, weight, emissions, methane, water, energy, land_use):
    """Save food waste data to CSV."""
    # Convert volume from m³ to cm³
    volume_cm3 = volume * 1_000_000
    
    df = pd.DataFrame([[timestamp, food_type, volume_cm3, density, weight, emissions, methane, water, energy, land_use]],
                      columns=["Timestamp", "Food Type", "Volume (cm³)", "Density (kg/m³)", "Weight (kg)",
                               "CO₂ Emissions (kg)", "Methane Emissions (kg)", "Water Usage (L)",
                               "Energy Usage (MJ)", "Land Use (Hectares)"])
    df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
    print(f"📄 Data saved to {CSV_FILE_PATH}")
  
def detect_food_objects(image):
    """Detect food objects in the frame and return bounding boxes."""
    if image is None or image.size == 0:
        print("⚠️ No valid image received for object detection.")
        return []

    # Create a copy of the image for drawing
    debug_image = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    min_area = 500  # Minimum area to consider as a valid object
    max_area = image.shape[0] * image.shape[1] * 0.5  # Maximum area (50% of frame)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get rotated rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # Using np.intp instead of np.int0
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w)/h if h > 0 else 0
            
            # Filter based on aspect ratio (typical food items are not extremely elongated)
            if 0.2 < aspect_ratio < 5:
                bounding_boxes.append((x, y, w, h))
                
                # Draw bounding box on debug image
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return bounding_boxes

@app.route('/latest_waste')
def latest_waste():
    try:
        with waste_data_lock:
            data = current_waste_data.copy()
        
        # Calculate derived metrics
        trees_lost = round(data["total_co2"] / TREES_ABSORPTION_CO2, 2)
        showers_wasted = round(data["total_water"] / 50, 1)  # Avg. shower water use in liters
        laptop_hours = round(data["total_energy"] / 0.05, 1)  # Energy usage per hour of laptop usage
        
        return jsonify({
            "latest_food": data["latest_food"],
            "latest_weight": round(data["total_weight"], 2),
            "latest_CO2": round(data["total_co2"], 2),
            "methane_emissions": round(data["total_methane"], 2),
            "water_usage": round(data["total_water"], 2),
            "energy_usage": round(data["total_energy"], 2),
            "trees_lost": trees_lost,
            "showers_wasted": showers_wasted,
            "laptop_hours": laptop_hours,
            "goal_message": data["goal_message"]
        })

    except Exception as e:
        print(f"⚠️ Error in latest_waste route: {str(e)}")
        return jsonify({"error": str(e)})

def get_food_density_usda(food_type):
    """Get food density from USDA database or use default values."""
    # This is a simplified version - in production, you'd want to use a proper database
    density_map = {
        "rice": 850,
        "pasta": 800,
        "bread": 300,
        "cheese": 1000,
        "pizza": 400,
        "meat": 1000,
        "vegetables": 700,
        "fruits": 800,
        "bones": 1200,
        "cereal": 400,
        "eggshells": 1000,
        "fish": 1000,
        "nutshells": 800,
        "onion-peels": 200,
        "potatoes": 700,
        "tomatoes": 700,
        "carrots": 700,
        "corncobs": 400,
        "watermelon-rind": 300,
        "grapes": 600,
        "lettuce": 200,
        "unknown": DEFAULT_DENSITY
    }
    return density_map.get(food_type.lower(), DEFAULT_DENSITY)

def detect_motion(prev_frame, current_frame, threshold=15, min_area=1000):
    """Detect motion between frames with improved sensitivity."""
    if prev_frame is None or current_frame is None:
        return False, 0

    # Apply Gaussian Blur to reduce noise
    prev_frame_blur = cv2.GaussianBlur(prev_frame, (5, 5), 0)
    current_frame_blur = cv2.GaussianBlur(current_frame, (5, 5), 0)

    # Compute absolute difference between frames
    diff = cv2.absdiff(prev_frame_blur, current_frame_blur)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours of motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate total motion area
    motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
    
    # Calculate motion score (normalized by frame size)
    frame_area = prev_frame.shape[0] * prev_frame.shape[1]
    motion_score = (motion_area / frame_area) * 100
    
    return motion_score > threshold, motion_score

@app.route('/reset_session')
def reset_session():
    """Manually reset the current session."""
    global manual_reset
    manual_reset = True
    return jsonify({"status": "Session reset initiated"})

def reset_session_data():
    """Reset all session data."""
    global manual_reset
    
    # Reset shared data with lock
    with waste_data_lock:
        current_waste_data["total_weight"] = 0
        current_waste_data["total_co2"] = 0
        current_waste_data["total_methane"] = 0
        current_waste_data["total_water"] = 0
        current_waste_data["total_energy"] = 0
        current_waste_data["total_land_use"] = 0
        current_waste_data["latest_food"] = "No waste detected yet"
        current_waste_data["goal_message"] = "Help us reach our goal!"
    
    manual_reset = False
    print("🔄 Session data reset")

# Add after imports
class RealSenseError(Exception):
    """Custom exception for RealSense camera errors."""
    pass

def initialize_realsense():
    """Initialize RealSense camera with proper error handling."""
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start pipeline
        pipeline.start(config)
        
        # Wait for first frame to ensure camera is working
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            raise RealSenseError("Failed to get initial frames")
            
        return pipeline
        
    except rs.error as e:
        raise RealSenseError(f"RealSense error: {str(e)}")
    except Exception as e:
        raise RealSenseError(f"Failed to initialize RealSense: {str(e)}")

def main():
    last_motion_time = datetime.datetime.now()
    prev_frame = None
    
    try:
        # Initialize RealSense
        pipeline = initialize_realsense()
        print("✅ RealSense camera initialized successfully")
                
        while True:
            try:
                now = datetime.datetime.now()
                
                # Check for manual reset or timeout
                if manual_reset or (datetime.datetime.now() - last_motion_time).total_seconds() > SESSION_TIMEOUT:
                    print("⚠️ Resetting session data...")
                    reset_session_data()
                    last_motion_time = datetime.datetime.now()
                
                # Get frames with timeout
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print("⚠️ No frames received. Skipping this loop iteration.")
                    continue
                    
                color_image = np.asanyarray(color_frame.get_data())
                if color_image is None or color_image.size == 0:
                    print("⚠️ No valid color image received. Skipping processing.")
                    continue
                    
                # Convert current frame to grayscale for motion detection
                current_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # Detect motion
                motion_detected, motion_score = detect_motion(prev_frame, current_frame)
                
                if motion_detected:
                    last_motion_time = datetime.datetime.now()
                    print(f"🔴 Motion detected! Motion Score: {motion_score:.2f}")

                    detected_bboxes = detect_food_objects(color_image)
                    
                    for bbox in detected_bboxes:
                        x, y, w, h = bbox
                        cropped_food = color_image[y:y+h, x:x+w]

                        if cropped_food is not None and cropped_food.size > 0:
                            # Convert image to base64 for API
                            _, buffer = cv2.imencode('.jpg', cropped_food)
                            base64_image = base64.b64encode(buffer).decode('utf-8')
                            
                            # Get classification and environmental data for all items
                            detected_items = classify_food(base64_image)
                            
                            # Process each detected item
                            for food_type, density, co2_emission, methane_emission, water_usage, energy_usage in detected_items:
                                print(f"🟢 Detected Food: {food_type} at {bbox}")

                                # Calculate volume and weight
                                volume = get_food_volume(depth_frame, bbox)
                                weight = estimate_weight(depth_frame, bbox, food_type)
                                
                                # Calculate environmental impacts
                                co2_emissions = co2_emission * weight
                                methane_emissions = methane_emission * weight
                                water_usage = water_usage * weight
                                energy_usage = energy_usage * weight
                                methane_after_decay = methane_decay(methane_emissions, years=10)
                                land_use = weight * 0.1  # Simplified land use calculation
                                
                                # Update shared data with lock
                                with waste_data_lock:
                                    current_waste_data["total_weight"] += weight
                                    current_waste_data["total_co2"] += co2_emissions
                                    current_waste_data["total_methane"] += methane_after_decay
                                    current_waste_data["total_water"] += water_usage
                                    current_waste_data["total_energy"] += energy_usage
                                    current_waste_data["total_land_use"] += land_use
                                    current_waste_data["latest_food"] = food_type
                                
                                save_to_csv(now, food_type, volume, density, weight, co2_emissions, 
                                          methane_after_decay, water_usage, energy_usage, land_use)

                                print(f"🟢 Total Waste So Far - Weight: {current_waste_data['total_weight']:.2f} kg | CO₂: {current_waste_data['total_co2']:.2f} kg | Methane: {current_waste_data['total_methane']:.2f} kg")

                prev_frame = current_frame

            except rs.error as e:
                print(f"⚠️ RealSense error in main loop: {str(e)}")
                # Try to recover by reinitializing the camera
                try:
                    pipeline.stop()
                    pipeline = initialize_realsense()
                    print("✅ RealSense camera reinitialized successfully")
                except RealSenseError as e:
                    print(f"❌ Failed to recover RealSense camera: {str(e)}")
                    break
            except Exception as e:
                print(f"⚠️ Error in main loop: {str(e)}")
                continue
                
    except RealSenseError as e:
        print(f"❌ Fatal RealSense error: {str(e)}")
    finally:
        try:
            pipeline.stop()
        except:
            pass

if __name__ == "__main__":
    # Start Flask in a separate thread
    from threading import Thread
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Run the main processing loop
    main()
