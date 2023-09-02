FROM public.ecr.aws/lambda/python:3.10

COPY ./requirements.txt ${LAMBDA_TASK_ROOT}

RUN python3.10 -m pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python3.10 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY ./app.py ./helpers.py ./model.pt ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]