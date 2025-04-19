import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from models import db, User, Interaction
from model.predict import predict  # our toy model stub

# ── App setup ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
jwt = JWTManager(app)

# ── Auth endpoints ─────────────────────────────────────────────────────
@app.post("/api/register")
def register():
    data = request.json or {}
    if User.query.filter_by(email=data.get('email')).first():
        return {"msg": "exists"}, 400
    user = User(
        email=data['email'],
        pw_hash=generate_password_hash(data['password'])
    )
    db.session.add(user)
    db.session.commit()
    return {"msg": "ok"}, 201

@app.post("/api/login")
def login():
    data = request.json or {}
    user = User.query.filter_by(email=data.get('email')).first()
    if not user or not check_password_hash(user.pw_hash, data['password']):
        return {"msg": "bad"}, 401
    token = create_access_token(identity=user.id)
    return {"access_token": token}

# ── Prediction endpoint ─────────────────────────────────────────────────
@app.post("/api/predict")
@jwt_required()
def predict_route():
    uid = get_jwt_identity()
    text = request.form.get('text')
    file = request.files.get('image')

    img_path = None
    if file:
        fname = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(save_path)
        img_path = save_path

    label, precautions = predict(text=text, image_path=img_path)

    rec = Interaction(
        user_id=uid,
        text=text,
        image_path=img_path,
        predicted_label=label,
        precautions=precautions
    )
    db.session.add(rec)
    db.session.commit()

    return jsonify(label=label, precautions=precautions)

# ── Serve uploaded images ───────────────────────────────────────────────
@app.get("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
