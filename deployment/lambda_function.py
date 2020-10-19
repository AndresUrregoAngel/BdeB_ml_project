import json
import logging.config
import boto3 

logging.config.fileConfig('logging.ini',disable_existing_loggers=False)
log = logging.getLogger(__name__)


def lambda_handler(event, context):
    # TODO implement
    sgmkcli = boto3.client('sagemaker-runtime')
    endpoint_name = [{"model":"LR","endpoint":'mlr-scikit-endpoint'} ,{"model":"Decision Tree","endpoint": 'mdtree-scikit-endpoint'},{"model":"GMM","endpoint":'mgmm-scikit-endpoint'}]
    
    age = event['queryStringParameters']['age']
    anemia = event['queryStringParameters']['anemia']
    creatinine = event['queryStringParameters']['creatinine']
    diabetes = event['queryStringParameters']['diabetes']
    ejection_fraction = event['queryStringParameters']['ejection_fraction']
    high_blood_pressure = event['queryStringParameters']['high_blood_pressure']
    platelets = event['queryStringParameters']['platelets']
    serum_creatinina = event['queryStringParameters']['serum_creatinina']
    serum_sodioum = event['queryStringParameters']['serum_sodioum']
    sex = event['queryStringParameters']['sex']
    smoking = event['queryStringParameters']['smoking']
    time = event['queryStringParameters']['time']
    
    body = f"{age},{anemia},{creatinine},{diabetes},{ejection_fraction},{high_blood_pressure},{platelets},{serum_creatinina},{serum_sodioum},{sex},{smoking},{time}"
    log.info(f"payload received {body}")
    
    log.info("Sending request to the ML endpoint")
    output = []
    for model in endpoint_name:
        response = sgmkcli.invoke_endpoint(
            EndpointName=model['endpoint'],  
            ContentType='text/csv',
                Body= body
            )
    
        result = response['Body']
        model_prediction = json.loads(result.read())
        output_mmsg = json.dumps(
            {f"the prediciton over the model {model['model']} is": f"likelihood of patients survive YES : {model_prediction['instances'][0]['features'][0]}, NO: {model_prediction['instances'][0]['features'][1]}"})
        output.append(output_mmsg)
    
    
    log.info(f"the result to be sent back is :{output}")
    
    
    return {
        'statusCode': 200,
        'body': json.dumps(output)
    }
