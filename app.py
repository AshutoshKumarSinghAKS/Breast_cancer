from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Feature mappings (based on how your model was trained)
AGE_MAPPING = {'20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}
MENOPAUSE_MAPPING = {'lt40': 0, 'ge40': 1, 'premeno': 2}
INV_NODES_MAPPING = {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '24-26': 6}
NODE_CAPS_MAPPING = {'no': 0, 'yes': 1}
BREAST_MAPPING = {'left': 0, 'right': 1}
BREAST_QUAD_MAPPING = {'left_up': 0, 'left_low': 1, 'right_up': 2, 'right_low': 3, 'central': 4}
IRRADIATE_MAPPING = {'no': 0, 'yes': 1}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get all form values
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)

        output = "Malignant" if prediction[0] == 1 else "Benign"
        return render_template('result.html', prediction_text=f'The tumor is {output}')
    except Exception as e:
        return jsonify({'error': str(e)})

# Run app
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
