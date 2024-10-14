from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and data once when the server starts
user_to_product_interaction = joblib.load('model/user_to_product_interaction.pkl')
item_list = joblib.load('model/item_list.pkl').tolist()
product_to_feature_interaction = joblib.load('model/product_to_feature.pkl')
user_to_index_mapping = joblib.load('model/user_to_index_mapping.pkl')
model = joblib.load('model/trained_model.pkl')

@app.route('/')
def index():
    # Render the homepage with the list of items for the user to select
    return render_template('index.html', items=item_list[:35])  # Display the first 35 items as an example

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    selected_items = data.get('selected_items', [])
    user_id = data.get('user_id', None)

    if user_id:
        # If a user ID is provided, generate recommendations for the user
        userindex = user_to_index_mapping.get(int(user_id), None)
        if userindex is None:
            return jsonify({'error': 'User ID not found', 'recommended_items': []})

        scores = model.predict(
            user_ids=userindex,
            item_ids=np.arange(len(item_list)),
            item_features=product_to_feature_interaction,
            user_features=None
        )
        recommended_items = [item_list[i] for i in np.argsort(-scores)][:10]
         
    else:
        # Otherwise, generate recommendations based on the selected items
        selected_indices = [item_list.index(item) for item in selected_items if item in item_list]
        user_interaction_vector = np.zeros(len(item_list))
        for index in selected_indices:
            user_interaction_vector[index] = 1

        scores = model.predict(
            user_ids=0,  # Using a temporary new user ID
            item_ids=np.arange(len(item_list)),
            item_features=product_to_feature_interaction,
            user_features=None
        )

        recommended_items = [item_list[i] for i in np.argsort(-scores) if i not in selected_indices][:10]

    return jsonify({'recommended_items': recommended_items})

if __name__ == '__main__':
    app.run(debug=True)

