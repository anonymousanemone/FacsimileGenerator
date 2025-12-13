import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from facsimile.processing import process_pipeline

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/facsimile')
def facsimile():
    return render_template('facsimile.html')

@app.route("/facsimile/process", methods=["POST"])
def process():
    if "images" not in request.files:
        flash("No images uploaded")
        return redirect(url_for("index"))
    
    files = request.files.getlist("images")
    saved_files = request.form.getlist("saved_files")
    saved_paths = []
    for f in files:
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            unique_name = f"{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            f.save(path)
            saved_paths.append(path)
    for fname in saved_files:
        path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        if os.path.exists(path):
            saved_paths.append(path)

    if not saved_paths:
        flash("No valid image files uploaded")
        return redirect(url_for("index"))

    # Collect options from form
    opts = {
        "binarize": {
            "method": request.form.get("bin_method", "WOLF"),
            "bilateral": {
                "d": int(request.form.get("seg_bilat_d", 5)),
                "sigma_color": int(request.form.get("bin_bilat_sc", 30)),
                "sigma_space": int(request.form.get("bin_bilat_ss", 5)),
            },
            "median_k": int(request.form.get("bin_med", 3)),
            "morph_open_k": int(request.form.get("bin_open", 3)),
            "morph_close_k": int(request.form.get("bin_close", 10)),
        },
        "segment": {
            "method": request.form.get("seg_method", "color"),
            "bilateral": {
                "d": int(request.form.get("seg_bilat_d", 5)),
                "sigma_color": int(request.form.get("seg_bilat_sc", 50)),
                "sigma_space": int(request.form.get("seg_bilat_ss", 7)),
            },
            "morph_close_k": int(request.form.get("seg_close", 15))
        },
        "overlay": bool(request.form.get("overlay", False))
    }

    try:
        output_name = process_pipeline(saved_paths, opts, app.config["OUTPUT_FOLDER"])
    except Exception as e:
        flash(f"Processing failed: {e}")
        return redirect(url_for("index"))

    # output into a certain window
    uploaded_filenames = [os.path.basename(p) for p in saved_paths]
    return render_template(
        "facsimile.html",
        result_image=url_for("static", filename=f"outputs/{output_name}"),
        uploaded_files=uploaded_filenames,
        opts=opts
    )


@app.route('/stitching')
def stitching():
    return render_template('stitching.html')


@app.route('/reconstruction')
def reconstruction():
    return render_template('reconstruction.html')


if __name__ == '__main__':
    app.run(debug=True)
