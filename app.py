import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from facsimile.testing import process_pipeline  # ← Import the processing pipeline
import facsimile.src.segmentation as segmentation

# -------------------------------
# Flask setup
# -------------------------------
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXT = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.secret_key = "replace-me-with-a-secret"

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in ALLOWED_EXT   
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/facsimile')
def facsimile():
    return render_template('facsimile.html')

@app.route("/process", methods=["POST"])
def process():
    if "images" not in request.files:
        flash("No images uploaded")
        return redirect(url_for("index"))
    
    files = request.files.getlist("images")
    saved_paths = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            f.save(path)
            saved_paths.append(path)

    if not saved_paths:
        flash("No valid image files uploaded")
        return redirect(url_for("index"))

    # Collect options from form
    opts = {
        "morph_close": request.form.get("morph_kernel", 3),
        "binarize_method": request.form.get("binarize_method", "WOLF"),
        "denoise_gaus": request.form.get("denoise_gaus", 10),
        "denoise_med": request.form.get("denoise_med", 10),
        "overlay": request.form.get("overlay") == "on",
    }

    try:
        output_name = process_pipeline(saved_paths, opts, app.config["OUTPUT_FOLDER"])
    except Exception as e:
        flash(f"Processing failed: {e}")
        return redirect(url_for("index"))

    # output into a certain window
    return redirect(url_for("result", filename=output_name))


@app.route('/stitching')
def stitching():
    return render_template('stitching.html')


@app.route('/reconstruction')
def reconstruction():
    return render_template('reconstruction.html')


if __name__ == '__main__':
    app.run(debug=True)

# -------------------------------
# Routes
# -------------------------------
"""
@app.route("/", methods=["GET"])
def index():
    config = segmentation.load_color_config()
    colors = config["color_ranges"]
    return render_template("index.html", colors=colors)

@app.route("/add", methods=["POST"])
def add_color():
    config = segmentation.load_color_config()

    new_entry = {
        "name": request.form["name"],
        "lower": [int(request.form["l_h"]), int(request.form["l_s"]), int(request.form["l_v"])],
        "upper": [int(request.form["u_h"]), int(request.form["u_s"]), int(request.form["u_v"])]
    }

    config["color_ranges"].append(new_entry)
    segmentation.save_color_config(config)

    return redirect("/")


@app.route("/delete/<name>")
def delete_color(name):
    config = segmentation.load_color_config()
    config["color_ranges"] = [c for c in config["color_ranges"] if c["name"] != name]
    segmentation.save_color_config(config)

    return redirect("/")


@app.route("/update/<name>", methods=["POST"])
def update_color(name):
    config = segmentation.load_color_config()

    for c in config["color_ranges"]:
        if c["name"] == name:
            c["lower"] = [int(request.form["l_h"]), int(request.form["l_s"]), int(request.form["l_v"])]
            c["upper"] = [int(request.form["u_h"]), int(request.form["u_s"]), int(request.form["u_v"])]
            break

    segmentation.save_color_config(config)
    return redirect("/")


@app.route("/process", methods=["POST"])
def process():
    if "images" not in request.files:
        flash("No images uploaded")
        return redirect(url_for("index"))
    
    files = request.files.getlist("images")
    saved_paths = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            f.save(path)
            saved_paths.append(path)

    if not saved_paths:
        flash("No valid image files uploaded")
        return redirect(url_for("index"))

    # Collect options from form
    opts = {
        "morph_close": request.form.get("morph_kernel", 3),
        "binarize_method": request.form.get("binarize_method", "WOLF"),
        "denoise_gaus": request.form.get("denoise_gaus", 10),
        "denoise_med": request.form.get("denoise_med", 10),
        "edge_overlay": request.form.get("edge_overlay") == "on",
    }

    try:
        output_name = process_pipeline(saved_paths, opts, app.config["OUTPUT_FOLDER"])
    except Exception as e:
        flash(f"Processing failed: {e}")
        return redirect(url_for("index"))

    return redirect(url_for("result", filename=output_name))

@app.route("/result/<filename>")
def result(filename):
    return render_template("result.html", filename=filename)

@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
"""