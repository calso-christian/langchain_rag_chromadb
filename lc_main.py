from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse 
from langserve import add_routes

import lc_question_helper

app = FastAPI()

@app.get('/')
def home_page():
    pass

chain=lc_question_helper.create_chain()

add_routes(
    app,
    chain,
    path='/chain'
)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8100)