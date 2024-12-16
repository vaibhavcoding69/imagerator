from flask import Flask, request, render_template, send_file, url_for
from huggingface_hub import InferenceClient
import io
import hashlib
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

client = InferenceClient("stabilityai/stable-diffusion-3.5-large", token="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")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image = client.text_to_image(prompt)
        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        # Save the image to a file on the server
        filename = str(hashlib.md5(prompt.encode()).hexdigest()) + '.png'
        file_path = os.path.join(app.static_folder, 'generated_images', filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            file.write(img_io.getvalue())
        # Pass the image URL to the template
        image_url = url_for('static', filename=f'generated_images/{filename}')
        return render_template('index.html', image_url=image_url)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)