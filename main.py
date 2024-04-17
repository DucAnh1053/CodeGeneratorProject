from fastapi import FastAPI, HTTPException
from encoder import Encoder
from attention import MultiHeadAttentionLayer
from decoder import Decoder
from model import MyModel, augment_tokenize_python_code
import uvicorn

import warnings
warnings.filterwarnings('ignore')


model = MyModel(model_path="model.pt", src_path="src.pt", trg_path="trg.pt")
# print("Model đã load thành công")
app = FastAPI()


@app.post("/")
def generate(msg: str):
    try:
        result = model.predict(msg)
        return {"result": result}
    except Exception as e:
        return HTTPException(status_code=404, detail="Đã có lỗi xảy ra")


if __name__ == "__main__":
    uvicorn.run("main:app", log_level="info")
