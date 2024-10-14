import joblib
import numpy as np
from scipy.sparse import coo_matrix # for constructing sparse matrix

def get_recommendations(model,user,items,user_to_product_interaction_matrix,user2index_map,product_to_feature_interaction_matrix):
# getting the userindex
    userindex = user2index_map.get(user, None)
    if userindex == None:
        return None
    users = userindex
    # Now perform the lookup using the CSR matrix
    known_positives = items[user_to_product_interaction_matrix.tocsr()[userindex].indices]
    print('User index =',users)
    # scores from model prediction
    scores = model.predict(user_ids = users, item_ids = np.arange(user_to_product_interaction_matrix.shape[1]),item_features=product_to_feature_interaction_matrix)
    #getting top items
    top_items = items[np.argsort(-scores)]
    # printing out the result
    print("User %s" % user)
    print(" Known positives:")
    for x in known_positives[:10]:
        print(" %s" % x)
    print(" Recommended:")
    for x in top_items[:10]:
        print(" %s" % x)

def get_recommendations_for_new_user(model, selected_items, items, product_to_feature_interaction_matrix):
    # Step 1: Create a temporary interaction vector for the new user
    num_items = len(items)
    user_interaction_vector = np.zeros(num_items)
    print("Items selected by the new user:")
    for item_index in selected_items:
        print(f" - {items[item_index]}")
    # Mark the selected items with a value indicating the user's preference (e.g., 1 for liked items)
    for item_index in selected_items:
        user_interaction_vector[item_index] = 1

    # Step 2: Use the interaction vector to predict scores for all items
    scores = model.predict(
        user_ids=0,  # Using user ID 0 for the temporary new user
        item_ids=np.arange(num_items),
        item_features=product_to_feature_interaction_matrix,
        user_features=None
    )

    # Step 3: Sort the items based on the predicted scores
    top_items = items[np.argsort(-scores)]

    # Step 4: Display the recommendations
    print("Recommended items for the new user:")
    for x in top_items[:10]:
        print(f" {x}")

# Example usage
if __name__ == "__main__":
    # Load the interaction matrix and other required data structures
    user_to_product_interaction = joblib.load('model/user_to_product_interaction.pkl')
    item_list = joblib.load('model/item_list.pkl')
    user_to_index_mapping = joblib.load('model/user_to_index_mapping.pkl')
    product_to_feature_interaction = joblib.load('model/product_to_feature.pkl')
    # Load the trained model
    model = joblib.load('model/trained_model.pkl')
    print("Model loaded successfully.")
    # Example: Assuming you have the necessary data loaded into variables
    # Call the recommendation function
    selected_items = [10, 23, 45, 11, 2, 9]  # Indices of items the new user has selected
    get_recommendations_for_new_user(model, selected_items, item_list, product_to_feature_interaction)
    # recommendations = get_recommendations(
    #     model,
    #     17017,
    #     item_list,
    #     user_to_product_interaction,
    #     user_to_index_mapping,
    #     product_to_feature_interaction
    # )
