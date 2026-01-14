import kfp
from kfp import dsl

@dsl.pipeline(
    name='Product Ingestion Pipeline',
    description='Pipeline to ingest products into Milvus'
)
def product_ingestion_pipeline():
    """
    A simple Kubeflow pipeline that executes the ingestion service.
    This pipeline assumes:
    1. The container image 'product-recommendation-server:latest' is built from server/Dockerfile.
    2. The 'documents' folder containing JSON data is mounted to /app/documents.
    """
    
    # Define the ingestion task
    # We execute the ingestion method directly using the server's codebase.
    ingest_task = dsl.ContainerOp(
        name='ingest_products',
        image='product-recommendation-server:latest',
        command=['python', '-c'],
        arguments=[
            'from services.ingestion_service import ingestion_service; '
            'print("Starting Ingestion Pipeline Task..."); '
            'ingestion_service.ingest_products()'
        ]
    )
    
if __name__ == '__main__':
    try:
        # Compile the pipeline
        kfp.compiler.Compiler().compile(
            pipeline_func=product_ingestion_pipeline,
            package_path='product_ingestion_pipeline.yaml'
        )
        print("Successfully compiled pipeline to product_ingestion_pipeline.yaml")
    except Exception as e:
        print(f"Failed to compile pipeline: {e}")
