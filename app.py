from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from routes import router
from descargar_modelos import descargar_modelo

app = FastAPI()
BUCKET = "mis-modelos-tesis"
descargar_modelo(BUCKET, 'modelos/randomForest_problematica/', 'modelos/randomForest_problematica')

# Configuración CORS para permitir conexión desde el frontend en Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://bertdirection.s3-website.us-east-2.amazonaws.com"],  # Puerto del frontend de Angular
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluimos las rutas que definimos en routes.py
app.include_router(router) 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
