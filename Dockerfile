FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Instalar torchaudio y libs de audio
# Instalar torchaudio y libs de audio
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Directorio de trabajo
WORKDIR /workspace

# Copiar el proyecto
COPY . /workspace

# Asegurar carpetas (aunque main.py también las crea)
RUN mkdir -p /workspace/data /workspace/results

# Comando por defecto: entrenar
CMD ["python", "main.py"]
