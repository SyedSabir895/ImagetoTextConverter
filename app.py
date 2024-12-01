from flask import Flask, request, render_template, jsonify
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize the model, tokenizer, and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Prediction function
def predict_step(image_path):
    images = []
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

# Route for uploading an image and getting prediction
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file is present
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if image_file:
            # Save the image temporarily
            image_path = os.path.join("uploads", image_file.filename)
            os.makedirs("uploads", exist_ok=True)
            image_file.save(image_path)

            # Get the prediction
            try:
                caption = predict_step(image_path)
            except Exception as e:
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
            finally:
                # Clean up the uploaded file
                if os.path.exists(image_path):
                    os.remove(image_path)

            return render_template('result.html', caption=caption)

    # Render upload form
    return render_template('upload.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
