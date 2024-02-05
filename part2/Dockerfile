FROM public.ecr.aws/lambda/python:3.11

# copy necessary files
COPY lambdacodes.py ./
COPY requirements.txt ./
COPY ./gpt_store/ ./gpt_store

RUN python3 -m pip install -r requirements.txt

# run lambda handler so that external service such as Lex can access it
CMD ["lambdacodes.main"]