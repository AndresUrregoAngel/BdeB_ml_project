import json
import logging.config
import boto3 

logging.config.fileConfig('logging.ini',disable_existing_loggers=False)
log = logging.getLogger(__name__)


def lambda_handler(event, context):
    # TODO implement
    sgmkcli = boto3.client('sagemaker-runtime')
    endpoint_name = 'mlr-scikit-endpoint'
    body = event['queryStringParameters']['features']
    
    log.info("Sending request to the ML endpoint")
    response = sgmkcli.invoke_endpoint(
        EndpointName=endpoint_name,  
        ContentType='text/csv',
            Body= body
        )

    result = response['Body']
    model_prediction = json.loads(result.read())
    
    output_mmsg = json.dumps({'LR prediction': model_prediction['instances'][0]['features']})
    
    
    return {
        'statusCode': 200,
        'body': output_mmsg
    }
