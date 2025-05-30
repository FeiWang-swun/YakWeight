import json
import os
import cv2
import numpy as np

# Iterate through all JSON files in the 'jsons' directory
for fname in os.listdir("jsons"):
    print(fname)

    # Open and load the JSON file
    with open(f"./jsons/{fname}", "r", encoding='utf-8') as f:
        data = json.load(f)

    # Initialize dictionary to store cow information
    cows = {"cows": []}
    other_points = []

    # Process each label in the JSON data
    for label in data["shapes"]:
        # Handle cow bounding boxes (2-point rectangles)
        if label["label"] == "cow" and len(label["points"]) == 2:
            # Extract and sort coordinates to get bounding box
            x_coords = [label["points"][0][0], label["points"][1][0]]
            y_coords = [label["points"][0][1], label["points"][1][1]]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)

            # Skip invalid boxes (zero width or height)
            if xmin == xmax or ymin == ymax:
                continue

            # Add cow bounding box to our dictionary
            cows["cows"].append({"cow": [xmin, ymin, xmax, ymax]})

        # Collect other points that aren't 'yak' labels
        elif label["label"] != "yak":
            other_points.append(label)

    # Only proceed if we found any cows
    if len(cows["cows"]) != 0:
        # Match points to their respective cow bounding boxes
        for cow in cows["cows"]:
            points_in_cow = {}

            # Check which points fall within this cow's bounding box
            for point in other_points:
                x, y = point["points"][0]
                if (cow["cow"][0] <= x <= cow["cow"][2] and
                        cow["cow"][1] <= y <= cow["cow"][3]):
                    points_in_cow[point["label"]] = point["points"]

            # Only include cows that have either 3 or 4 specific keypoints
            if len(points_in_cow) == 4:
                required_keys = {"back", "tail", "shoulder", "foot"}
                if required_keys.issubset(points_in_cow.keys()):
                    cow["point"] = points_in_cow
            elif len(points_in_cow) == 3:
                required_keys = {"back", "tail", "shoulder"}
                if required_keys.issubset(points_in_cow.keys()):
                    cow["point"] = points_in_cow

        # Add image metadata to our output
        cows["imageHeight"] = data["imageHeight"]
        cows["imageWidth"] = data["imageWidth"]

        # Try to find the corresponding image file
        img_name = fname.split(".")[0]
        img_path = f"images/{img_name}.jpg"
        img_original = cv2.imread(img_path)

        # Handle case where image might be PNG instead of JPG
        if not isinstance(img_original, np.ndarray):
            img_path = f"images/{img_name}.png"

        cows["imagePath"] = os.path.basename(img_path)

        # Save the processed data to new JSON file
        output_path = f"./kp_json4/{fname}"
        with open(output_path, 'w') as f:
            json.dump(cows, f)