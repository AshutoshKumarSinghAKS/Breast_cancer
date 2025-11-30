from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# -----------------------------
# Load model and scaler
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Mappings (aligned with training)
# -----------------------------
# Age was label-encoded on strings like "20-29", "30-39", etc.
# LabelEncoder encodes in sorted (lexicographic) order.
AGE_MAPPING = {
    "20-29": 0,
    "30-39": 1,
    "40-49": 2,
    "50-59": 3,
    "60-69": 4,
    "70-79": 5,
}

# You manually mapped menopause in training:
# "premeno": -1, "ge40": 0, "lt40": 1
MENOPAUSE_MAPPING = {
    "premeno": -1,
    "ge40": 0,
    "lt40": 1,
}

# inv-nodes was label-encoded on string ranges.
# LabelEncoder order = sorted classes:
# ['0-2','12-14','15-17','24-26','3-5','6-8','9-11']
INV_NODES_MAPPING = {
    "0-2": 0,
    "12-14": 1,
    "15-17": 2,
    "24-26": 3,
    "3-5": 4,
    "6-8": 5,
    "9-11": 6,
}

# node-caps was mapped manually: {'yes': 1, 'no': 0}
NODE_CAPS_MAPPING = {
    "no": 0,
    "yes": 1,
}

# breast was label-encoded on ["left","right"] → ['left','right'] -> [0,1]
BREAST_MAPPING = {
    "left": 0,
    "right": 1,
}

# breast-quad label-encoded on sorted classes:
# ['central','left_low','left_up','right_low','right_up']
BREAST_QUAD_MAPPING = {
    "central": 0,
    "left_low": 1,
    "left_up": 2,
    "right_low": 3,
    "right_up": 4,
}

# irradiate label-encoded: ['no','yes'] -> [0,1]
IRRADIATE_MAPPING = {
    "no": 0,
    "yes": 1,
}


# -----------------------------
# Preprocess function
# -----------------------------
def preprocess_input(form_data):
    """
    Convert raw form inputs into a numeric feature vector of length 9,
    in the same order used during training:
    ['age','menopause','tumer-size','inv-nodes',
     'node-caps','deg-malig','breast','breast-quad','irradiate']
    Then apply the saved StandardScaler.
    """
    try:
        age = AGE_MAPPING[form_data["age"]]
        menopause = MENOPAUSE_MAPPING[form_data["menopause"]]

        # In training, 'tumer-size' was numeric (mean of range).
        # Here we take direct mm input as float; it's fine.
        tumor_size = float(form_data["tumor_size"])

        inv_nodes = INV_NODES_MAPPING[form_data["inv_nodes"]]
        node_caps = NODE_CAPS_MAPPING[form_data["node_caps"]]
        deg_malig = float(form_data["deg_malig"])
        breast = BREAST_MAPPING[form_data["breast"]]
        breast_quad = BREAST_QUAD_MAPPING[form_data["breast_quad"]]
        irradiate = IRRADIATE_MAPPING[form_data["irradiate"]]

    except KeyError as e:
        raise ValueError(f"Invalid category value: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid numeric value: {e}")

    # Order MUST match training feature order
    features = np.array([
        age,
        menopause,
        tumor_size,
        inv_nodes,
        node_caps,
        deg_malig,
        breast,
        breast_quad,
        irradiate
    ], dtype=float).reshape(1, -1)

    # Your scaler was trained on ALL 9 features → transform full vector
    features_scaled = scaler.transform(features)

    return features_scaled


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = preprocess_input(request.form)
        pred = model.predict(X)[0]

        # In training, 'class' was label-encoded, so `pred` is 0/1
        if pred == 1:
            result = "High risk of recurrence (Cancerous)"
            alert_type = "danger"
            message = "Please consult an oncologist or healthcare professional for further evaluation."
        else:
            result = "Low risk of recurrence (Not Cancerous)"
            alert_type = "success"
            message = "Model predicts low risk, but always confirm with medical experts."

        return render_template(
            "index.html",
            prediction_text=result,
            alert_type=alert_type,
            message=message
        )

    except Exception as e:
        # For debugging in UI if something goes wrong
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('model.pkl', 'rb'))

# # Feature mappings (must match how the model was trained)
# AGE_MAPPING = {'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}
# MENOPAUSE_MAPPING = {'lt40': 0, 'ge40': 1, 'premeno': 2}
# INV_NODES_MAPPING = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26': 6}
# NODE_CAPS_MAPPING = {'no': 0, 'yes': 1}
# BREAST_MAPPING = {'left': 0, 'right': 1}
# BREAST_QUAD_MAPPING = {'left_up': 0, 'left_low': 1, 'right_up': 2, 'right_low': 3, 'central': 4}
# IRRADIATE_MAPPING = {'no': 0, 'yes': 1}


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         age = request.form['age']
#         menopause = request.form['menopause']
#         tumor_size = float(request.form['tumor_size'])
#         inv_nodes = request.form['inv_nodes']
#         node_caps = request.form['node_caps']
#         deg_malig = int(request.form['deg_malig'])
#         breast = request.form['breast']
#         breast_quad = request.form['breast_quad']
#         irradiate = request.form['irradiate']

#         # Convert categorical values to numeric using mappings
#         age_val = AGE_MAPPING.get(age)
#         menopause_val = MENOPAUSE_MAPPING.get(menopause)
#         inv_nodes_val = INV_NODES_MAPPING.get(inv_nodes)
#         node_caps_val = NODE_CAPS_MAPPING.get(node_caps)
#         breast_val = BREAST_MAPPING.get(breast)
#         breast_quad_val = BREAST_QUAD_MAPPING.get(breast_quad)
#         irradiate_val = IRRADIATE_MAPPING.get(irradiate)

#         # Combine all features into a single array (order must match training)
#         features = np.array([[age_val, menopause_val, tumor_size, inv_nodes_val, node_caps_val,
#                               deg_malig, breast_val, breast_quad_val, irradiate_val]])

#         # Make prediction
#         prediction = model.predict(features)[0]

#         # Interpret prediction
#         if prediction == 1:
#             result = "High risk of recurrence"
#             alert_type = "danger"
#             message = "The model predicts a possibility of breast cancer recurrence. Please consult your doctor for professional advice."
#         else:
#             result = "Low risk of recurrence"
#             alert_type = "success"
#             message = "The model predicts a low likelihood of breast cancer recurrence."

#         return render_template('index.html',
#                                prediction_text=result,
#                                alert_type=alert_type,
#                                message=message)

#     except Exception as e:
#         return jsonify({'error': str(e)})


# if __name__ == "__main__":
#     app.run(debug=True)
