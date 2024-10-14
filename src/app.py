from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and data once when the server starts
user_to_product_interaction = joblib.load('model/user_to_product_interaction.pkl')
item_list = joblib.load('model/item_list.pkl').tolist()
product_to_feature_interaction = joblib.load('model/product_to_feature.pkl')
model = joblib.load('model/trained_model.pkl')

@app.route('/')
def index():
    # Render the homepage with the list of items for the user to select
    return render_template('index.html', items=item_list[:35])  # Display the first 15 items as an example

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_items = request.json.get('selected_items', [])
    print(f"Selected items: {selected_items}")  # Debugging line
    # Get the selected items from the user
   # Convert the selected items to their indices in the item_list
    selected_indices = [item_list.index(item) for item in selected_items if item in item_list]
    
    # Generate recommendations using the selected items
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
