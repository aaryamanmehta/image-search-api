import os
import zipfile
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import vecs
import numpy as np
import argparse
from datetime import date
from flask import Flask, request, jsonify

DB_CONNECTION = "postgresql://postgres.flapymzejiswrwxiluxf:m49WVIWLTgPc5pqU@aws-0-us-east-1.pooler.supabase.com:5432/postgres"

app = Flask(__name__)

@app.route('/seed', methods=['POST'])
def seed():
    model_name = 'clip-ViT-B-32'
    dimension = 512
    collection_name = request.json.get('collection_name')
    zip_file = request.json.get('zip_file')

    try:
        # create vector store client
        vx = vecs.create_client(DB_CONNECTION)
        # Shorten the collection name to avoid long collection names. Take each first letter of each word in the collection name
        vector_collection_name = ''
        vector_collection_name += model_name
        for word in collection_name.split():
            vector_collection_name += word[0]
        # make vector collection name more unique by adding today' date
        vector_collection_name += str(date.today())
        # make vector collection name lowercase
        vector_collection_name = vector_collection_name.lower()
        # create a collection of vectors with the appropriate dimensions
        images = vx.get_or_create_collection(name=f"{vector_collection_name}", dimension=dimension)

        # Load the selected CLIP model
        model = SentenceTransformer(model_name)

        # Loop through the folders in the zip file
        with zipfile.ZipFile(zip_file) as zip_ref:
            for folder_name in zip_ref.namelist():
                print(f"Processing folder: {folder_name}")
                if folder_name.endswith('.png'):
                    with zip_ref.open(folder_name) as image_file:
                        img = Image.open(image_file)
                        img_emb = model.encode(img)
                        images.upsert(
                            records=[
                                (
                                    folder_name,        # the vector's identifier
                                    img_emb,          # the vector. list or np.array
                                    {"type": "png"}   # associated  metadata
                                )
                            ]
                        )
                        print(f"Inserted {folder_name}")

        print("Inserted images")

        # index the collection for fast search performance
        images.create_index(
            method=vecs.IndexMethod.auto,
            measure=vecs.IndexMeasure.cosine_distance
        )

        print("Created index")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    return jsonify({'message': 'Images seeded successfully'})


@app.route('/search', methods=['POST'])
def search():
    model_name = 'clip-ViT-B-32'
    dimension = 512
    query_string = request.json.get('query_string')
    similarity_threshold = request.json.get('similarity_threshold')
    limit = request.json.get('limit')
    zip_file = request.json.get('zip_file')
    collection_name = request.json.get('collection_name')
    final_return = []
    try:
        # create vector store client
        vx = vecs.create_client(DB_CONNECTION)
        vector_collection_name = ''
        vector_collection_name += model_name
        for word in collection_name.split():
            vector_collection_name += word[0]
        # make vector collection name more unique by adding today' date
        vector_collection_name += str(date.today())
        # make vector collection name lowercase
        vector_collection_name = vector_collection_name.lower()

        # create a collection of vectors with the appropriate dimensions
        images = vx.get_or_create_collection(name=f"{vector_collection_name}", dimension=dimension)

        # Load the selected CLIP model
        model = SentenceTransformer(model_name)
        # Encode text query
        text_emb = model.encode(query_string)

        # query the collection filtering metadata for "type" = "png"
        results = images.query(
            data=text_emb,  # required
            limit=limit,  # optional
            filters={"type": {"$eq": "png"}},  # metadata filters
        )
        print(results)

        displayed_images = []  # Store displayed images for download
        
        with zipfile.ZipFile(zip_file) as zip_ref:
            for result in results:
                # Load the image
                    with zip_ref.open(result) as image_file:
                        img = Image.open(image_file)
                        # Encode the image
                        img_emb = model.encode(img)
                        # Calculate similarity between query and image embeddings
                        similarity = np.dot(text_emb, img_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
                        print("Similarity:", similarity)
                        if similarity > similarity_threshold:
                            displayed_images.append(result)  # Add displayed images
                            
        final_return.append(displayed_images)

        if not results:
            print("No images found matching the query.")

        return displayed_images

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return jsonify({'message': 'Images searched successfully', 'images': final_return})



@app.route('/auto_search', methods=['POST'])
def auto_search():
    model_name = 'clip-ViT-B-32'
    dimension = 512
    query_string = request.json.get('query_string')
    zip_file = request.json.get('zip_file')
    collection_name = request.json.get('collection_name')
    final_return = []
    try:
        # create vector store client
        vx = vecs.create_client(DB_CONNECTION)
        vector_collection_name = ''
        vector_collection_name += model_name
        for word in collection_name.split():
            vector_collection_name += word[0]
        # make vector collection name more unique by adding today' date
        vector_collection_name += str(date.today())
        # make vector collection name lowercase
        vector_collection_name = vector_collection_name.lower()

        # create a collection of vectors with the appropriate dimensions
        images = vx.get_or_create_collection(name=f"{vector_collection_name}", dimension=dimension)

        # Load the selected CLIP model
        model = SentenceTransformer(model_name)
        # Encode text query
        text_emb = model.encode(query_string)

        # query the collection filtering metadata for "type" = "png"
        results = images.query(
            data=text_emb,  # required
            filters={"type": {"$eq": "png"}},  # metadata filters
        )
        print(results)

        displayed_images = []  # Store displayed images for download
        with zipfile.ZipFile(zip_file) as zip_ref:
            for result in results:
                # Load the image
                    with zip_ref.open(result) as image_file:
                        img = Image.open(image_file)
                        img_emb = model.encode(img)
                        # Calculate similarity between query and image embeddings
                        similarity = np.dot(text_emb, img_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
                        displayed_images.append(result)  # Add displayed images

        final_return.append(displayed_images)
        if not results:
            print("No images found matching the query.")

        return displayed_images

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return jsonify({'message': 'Images searched successfully', 'images': final_return})

@app.route('/cluster_search', methods=['POST'])
def cluster_search():
    model_name = 'clip-ViT-B-32'
    dimension = 512
    cluster_similarity_threshold = 0.8
    min_community_size = 2
    max_community_size = 10
    zip_file = request.json.get('zip_file')
    collection_name = request.json.get('collection_name')
    final_return = []
    try:
        # create vector store client
        vx = vecs.create_client(DB_CONNECTION)
        vector_collection_name = ''
        vector_collection_name += model_name
        for word in collection_name.split():
            vector_collection_name += word[0]
        # make vector collection name more unique by adding today' date
        vector_collection_name += str(date.today())
        # make vector collection name lowercase
        vector_collection_name = vector_collection_name.lower()

        # create a collection of vectors with the appropriate dimensions
        images = vx.get_or_create_collection(name=f"{vector_collection_name}", dimension=dimension)

        # Load the selected CLIP model
        model = SentenceTransformer(model_name)

        # Compute similarity for all image embeddings
        results = images.query(
            data=None,  # required
            limit=309,  # optional
            filters={"type": {"$eq": "png"}},  # metadata filters
        )

        # Create a list of image embeddings
        image_embeddings = []
        with zipfile.ZipFile(zip_file) as zip_ref:
            for result in results:
                    with zip_ref.open(result) as image_file:
                        img = Image.open(image_file)
                        img_emb = model.encode(img)
                        image_embeddings.append(img_emb)
        # Convert to numpy ndarray
        image_embeddings = np.array(image_embeddings)

        duplicates = util.paraphrase_mining_embeddings(image_embeddings)

        cluster_data = []  # Store the cluster data in a specific format: cluster number, image names

        clusters = []

        # Group images into clusters
        for score, idx1, idx2 in duplicates:
            if score >= cluster_similarity_threshold:
                found = False
                for cluster in clusters:
                    if idx1 in cluster or idx2 in cluster:
                        cluster.add(idx1)
                        cluster.add(idx2)
                        found = True
                        break
                if not found:
                    clusters.append({idx1, idx2})

        # print(f"Found {len(clusters)} clusters")

        # Display images in clusters, with a label for each cluster,
        # if the cluster size is greater than the min_community_size and less than the max_community_size
        # if the cluster size is less than the min_community_size, do not display the cluster.
        # if the cluster size is greater than the max_community_size, do not display the cluster.
        for i, cluster in enumerate(clusters):
            if len(cluster) >= min_community_size and len(cluster) <= max_community_size:
                for idx in cluster:
                    result = results[idx]
                    cluster_data.append((i, result))
                    
        if not clusters:
            print("No clusters found.")
        print(cluster_data)
        final_return.append(cluster_data)
        return cluster_data

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return jsonify({'message': 'Images clustered successfully', 'images': final_return})

if __name__ == '__main__':
    app.run(host='0.0.0.0')