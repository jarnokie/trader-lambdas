FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN python3 -m pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY helpers.py ${LAMBDA_TASK_ROOT}
COPY model.pt ${LAMBDA_TASK_ROOT}
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]