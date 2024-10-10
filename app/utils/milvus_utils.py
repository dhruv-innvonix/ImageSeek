# app/utils/milvus_utils.py

from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    connections,
    utility
)

def get_collection_schema():
    """Define the schema for the Milvus collection."""
    fields = [
        FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # Adjust the dim based on your embedding size
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),  # Optional: to store category
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100, is_nullable=True),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500, is_nullable=True),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=200, is_nullable=True),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=100, is_nullable=True),
        FieldSchema(name="date_taken", dtype=DataType.VARCHAR, max_length=20, is_nullable=True),
        FieldSchema(name="uploader", dtype=DataType.VARCHAR, max_length=50, is_nullable=True),
        FieldSchema(name="quality_rating", dtype=DataType.FLOAT, is_nullable=True),
    ]
    return CollectionSchema(fields=fields, description="Image embeddings collection schema")

def create_collection_if_not_exists(collection_name: str):
    """Create a collection in Milvus if it does not exist."""
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Check if the collection exists using the utility module
    existing_collections = utility.list_collections()

    # for collection_name in existing_collections:
    #     # Create a Collection object for each collection name
    #     collection = Collection(collection_name)
        
    #     # Drop the collection
    #     collection.drop()

    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create the collection
        schema = get_collection_schema()
        Collection(collection_name, schema)
        print(f"Collection '{collection_name}' created.")
